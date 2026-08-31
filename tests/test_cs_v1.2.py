import argparse
import ctypes
import time
from pathlib import Path

import cv2
import mss
import numpy as np
from ultralytics import YOLO


# ============================================================
# 路径与类别
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "enemy_detector"
    / "weights"
    / "best.pt"
)

CLASS_NAMES = {
    0: "friendly_body",
    1: "enemy_body",
    2: "friendly_head",
    3: "enemy_head",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 255, 0),
    3: (0, 165, 255),
}

# 每组的躯干类别和头部类别
TEAM_PART_CLASSES = {
    "friendly": {
        "body": 0,
        "head": 2,
    },
    "enemy": {
        "body": 1,
        "head": 3,
    },
}


# ============================================================
# Windows 按键
# ============================================================

VK_RBUTTON = 0x02

VK_END = 0x23

VK_F1 = 0x70
VK_F2 = 0x71
VK_F3 = 0x72
VK_F4 = 0x73
VK_F5 = 0x74
VK_F6 = 0x75
VK_F7 = 0x76
VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79
VK_F11 = 0x7A

# [ 和 ]
VK_OEM_4 = 0xDB
VK_OEM_6 = 0xDD

MOUSEEVENTF_MOVE = 0x0001

user32 = ctypes.windll.user32


# ============================================================
# Windows 鼠标输入结构
# ============================================================

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", INPUT_UNION),
    ]


# ============================================================
# 基础工具
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def enable_dpi_awareness():
    """避免Windows显示缩放造成截屏坐标偏移。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def is_key_down(vk_code):
    return bool(user32.GetAsyncKeyState(vk_code) & 0x8000)


class HotkeyState:
    """将全局按键转换成单次按下事件。"""

    def __init__(self):
        self.previous = {}

    def pressed_once(self, vk_code):
        current = is_key_down(vk_code)
        previous = self.previous.get(vk_code, False)

        self.previous[vk_code] = current

        return current and not previous


# ============================================================
# 平滑鼠标控制器
# ============================================================

class AimController:
    """
    平滑相对鼠标移动。

    smoothing:
        0.00 = 无平滑，响应最快
        0.95 = 高平滑，移动更柔和
    """

    def __init__(self):
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0

        self.residual_x = 0.0
        self.residual_y = 0.0

        self.initialized = False

    def reset(self):
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0

        self.residual_x = 0.0
        self.residual_y = 0.0

        self.initialized = False

    def move(self, target_dx, target_dy, smoothing):
        smoothing = clamp(smoothing, 0.0, 0.95)

        # 平滑越高，新输入所占比例越低
        alpha = max(0.05, 1.0 - smoothing)

        if not self.initialized:
            self.smoothed_x = target_dx
            self.smoothed_y = target_dy
            self.initialized = True
        else:
            self.smoothed_x += (
                target_dx - self.smoothed_x
            ) * alpha

            self.smoothed_y += (
                target_dy - self.smoothed_y
            ) * alpha

        self.residual_x += self.smoothed_x
        self.residual_y += self.smoothed_y

        move_x = int(self.residual_x)
        move_y = int(self.residual_y)

        self.residual_x -= move_x
        self.residual_y -= move_y

        if move_x == 0 and move_y == 0:
            return

        mouse_input = MOUSEINPUT(
            dx=move_x,
            dy=move_y,
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE,
            time=0,
            dwExtraInfo=0,
        )

        input_data = INPUT(
            type=0,
            union=INPUT_UNION(mi=mouse_input),
        )

        user32.SendInput(
            1,
            ctypes.byref(input_data),
            ctypes.sizeof(INPUT),
        )


# ============================================================
# 参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO屏幕检测和单目标跟踪测试"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="YOLO权重路径",
    )

    parser.add_argument(
        "--roi-size",
        type=int,
        default=400,
        help="屏幕中心检测区域大小",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="检测置信度阈值",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU阈值",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="YOLO推理尺寸",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备，例如0或cpu；默认自动选择",
    )

    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="显示器编号",
    )

    parser.add_argument(
        "--x-strength",
        type=float,
        default=0.35,
        help="X轴移动强度",
    )

    parser.add_argument(
        "--y-strength",
        type=float,
        default=0.30,
        help="Y轴移动强度",
    )

    parser.add_argument(
        "--smooth",
        type=float,
        default=0.65,
        help="平滑强度，范围0.00～0.95",
    )

    parser.add_argument(
        "--aim-region",
        type=float,
        default=0.65,
        help="锁点区域：0为躯干中心，1为头部中心",
    )

    parser.add_argument(
        "--max-step",
        type=float,
        default=80.0,
        help="每帧单轴最大移动量",
    )

    parser.add_argument(
        "--deadzone",
        type=float,
        default=3.0,
        help="中心死区像素",
    )

    parser.add_argument(
        "--invert-y",
        action="store_true",
        help="反转Y轴",
    )

    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="启动时关闭调试窗口",
    )

    return parser.parse_args()


# ============================================================
# 截屏区域
# ============================================================

def calculate_center_roi(monitor, roi_size):
    roi_size = int(
        clamp(
            roi_size,
            100,
            min(monitor["width"], monitor["height"]),
        )
    )

    left = (
        monitor["left"]
        + (monitor["width"] - roi_size) // 2
    )

    top = (
        monitor["top"]
        + (monitor["height"] - roi_size) // 2
    )

    return {
        "left": int(left),
        "top": int(top),
        "width": roi_size,
        "height": roi_size,
    }


# ============================================================
# 检测结果
# ============================================================

def extract_detections(result):
    detections = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return detections

    coordinates = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.int().cpu().numpy()

    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().numpy()
    else:
        track_ids = [None] * len(boxes)

    for box, confidence, class_id, track_id in zip(
        coordinates,
        confidences,
        class_ids,
        track_ids,
    ):
        class_id = int(class_id)

        if class_id not in CLASS_NAMES:
            continue

        x1, y1, x2, y2 = box

        detections.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(confidence),
            "class_id": class_id,
            "track_id": (
                None
                if track_id is None
                else int(track_id)
            ),
        })

    return detections


def box_center(detection):
    return (
        (detection["x1"] + detection["x2"]) / 2.0,
        (detection["y1"] + detection["y2"]) / 2.0,
    )


def box_width(detection):
    return max(
        1.0,
        detection["x2"] - detection["x1"],
    )


def box_height(detection):
    return max(
        1.0,
        detection["y2"] - detection["y1"],
    )


def get_team_from_class(class_id):
    if class_id in (0, 2):
        return "friendly"

    if class_id in (1, 3):
        return "enemy"

    return None


# ============================================================
# 躯干与头部匹配
# ============================================================

def find_head_for_body(body, detections, head_class):
    """
    在躯干顶部附近寻找最可能属于该躯干的头部框。
    """

    body_center_x, _ = box_center(body)
    body_w = box_width(body)
    body_h = box_height(body)

    expected_x = body_center_x
    expected_y = body["y1"]

    candidates = []

    for detection in detections:
        if detection["class_id"] != head_class:
            continue

        head_x, head_y = box_center(detection)

        horizontal_margin = body_w * 0.25

        if not (
            body["x1"] - horizontal_margin
            <= head_x
            <= body["x2"] + horizontal_margin
        ):
            continue

        if not (
            body["y1"] - body_h * 0.45
            <= head_y
            <= body["y1"] + body_h * 0.40
        ):
            continue

        distance = (
            (head_x - expected_x) ** 2
            + (head_y - expected_y) ** 2
        )

        candidates.append((distance, detection))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def find_body_for_head(head, detections, body_class):
    """
    在头部下方寻找最可能属于该头部的躯干框。
    """

    head_x, head_y = box_center(head)
    head_h = box_height(head)

    candidates = []

    for detection in detections:
        if detection["class_id"] != body_class:
            continue

        body_x, _ = box_center(detection)
        body_w = box_width(detection)
        body_h = box_height(detection)

        horizontal_margin = body_w * 0.25

        if not (
            detection["x1"] - horizontal_margin
            <= head_x
            <= detection["x2"] + horizontal_margin
        ):
            continue

        if not (
            detection["y1"] - max(head_h * 2.0, body_h * 0.45)
            <= head_y
            <= detection["y1"] + body_h * 0.40
        ):
            continue

        expected_head_x = body_x
        expected_head_y = detection["y1"]

        distance = (
            (head_x - expected_head_x) ** 2
            + (head_y - expected_head_y) ** 2
        )

        candidates.append((distance, detection))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def get_body_and_head(anchor, detections):
    """
    根据当前锁定框寻找同一人物的躯干框和头部框。
    """

    team = get_team_from_class(anchor["class_id"])

    if team is None:
        return None, None

    body_class = TEAM_PART_CLASSES[team]["body"]
    head_class = TEAM_PART_CLASSES[team]["head"]

    if anchor["class_id"] == body_class:
        body = anchor
        head = find_head_for_body(
            body,
            detections,
            head_class,
        )

        return body, head

    head = anchor
    body = find_body_for_head(
        head,
        detections,
        body_class,
    )

    return body, head


def get_aim_point(anchor, detections, aim_region):
    """
    aim_region:
        0.00 = 躯干框中心
        1.00 = 头部框中心
        中间值 = 两个中心点之间线性插值
    """

    aim_region = clamp(aim_region, 0.0, 1.0)

    body, head = get_body_and_head(
        anchor,
        detections,
    )

    # 同时识别到躯干和头部
    if body is not None and head is not None:
        body_x, body_y = box_center(body)
        head_x, head_y = box_center(head)

    # 只有躯干：根据躯干框顶部估算头部中心
    elif body is not None:
        body_x, body_y = box_center(body)

        body_h = box_height(body)

        head_x = body_x
        head_y = body["y1"] + body_h * 0.08

    # 只有头部：根据头部大小估算躯干中心
    elif head is not None:
        head_x, head_y = box_center(head)

        head_h = box_height(head)

        body_x = head_x
        body_y = head_y + head_h * 2.2

    else:
        return box_center(anchor)

    aim_x = body_x + (head_x - body_x) * aim_region
    aim_y = body_y + (head_y - body_y) * aim_region

    return aim_x, aim_y


def select_target(
    detections,
    allowed_classes,
    center_x,
    center_y,
    aim_region,
):
    """
    恢复旧版锁敌逻辑：
    优先选择允许类别中的头部框；没有头部时选择躯干框；
    同类别中选择锁点距离准星最近的目标。
    """

    heads = [
        detection
        for detection in detections
        if (
            detection["class_id"] in allowed_classes
            and detection["class_id"] in (2, 3)
        )
    ]

    bodies = [
        detection
        for detection in detections
        if (
            detection["class_id"] in allowed_classes
            and detection["class_id"] in (0, 1)
        )
    ]

    candidates = heads if heads else bodies

    if not candidates:
        return None

    def target_distance(detection):
        aim_x, aim_y = get_aim_point(
            detection,
            detections,
            aim_region,
        )

        return (
            (aim_x - center_x) ** 2
            + (aim_y - center_y) ** 2
        )

    return min(candidates, key=target_distance)


# ============================================================
# 移动量计算
# ============================================================

def calculate_mouse_movement(
    target,
    detections,
    center_x,
    center_y,
    aim_region,
    x_strength,
    y_strength,
    max_step,
    deadzone,
    invert_y,
):
    aim_x, aim_y = get_aim_point(
        target,
        detections,
        aim_region,
    )

    error_x = aim_x - center_x
    error_y = aim_y - center_y

    if abs(error_x) <= deadzone:
        error_x = 0.0

    if abs(error_y) <= deadzone:
        error_y = 0.0

    move_x = error_x * x_strength
    move_y = error_y * y_strength

    if invert_y:
        move_y = -move_y

    move_x = clamp(
        move_x,
        -max_step,
        max_step,
    )

    move_y = clamp(
        move_y,
        -max_step,
        max_step,
    )

    return move_x, move_y, aim_x, aim_y


# ============================================================
# 调试画面
# ============================================================

def draw_debug(
    frame,
    detections,
    target,
    fps,
    roi_size,
    target_mode,
    x_strength,
    y_strength,
    smoothing,
    aim_region,
    aiming,
):
    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    for detection in detections:
        class_id = detection["class_id"]
        color = CLASS_COLORS[class_id]

        x1 = int(detection["x1"])
        y1 = int(detection["y1"])
        x2 = int(detection["x2"])
        y2 = int(detection["y2"])

        thickness = 3 if detection is target else 1

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        track_id = detection["track_id"]

        track_text = (
            "-"
            if track_id is None
            else str(track_id)
        )

        label = (
            f"{class_id} "
            f"{CLASS_NAMES[class_id]} "
            f"{detection['confidence']:.2f} "
            f"ID:{track_text}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )

    # 中心准星
    cv2.line(
        frame,
        (center_x - 12, center_y),
        (center_x + 12, center_y),
        (255, 255, 255),
        1,
    )

    cv2.line(
        frame,
        (center_x, center_y - 12),
        (center_x, center_y + 12),
        (255, 255, 255),
        1,
    )

    cv2.circle(
        frame,
        (center_x, center_y),
        3,
        (255, 255, 255),
        1,
    )

    if target is not None:
        aim_x, aim_y = get_aim_point(
            target,
            detections,
            aim_region,
        )

        aim_point = (
            int(aim_x),
            int(aim_y),
        )

        cv2.circle(
            frame,
            aim_point,
            7,
            (255, 0, 255),
            2,
        )

        cv2.line(
            frame,
            (center_x, center_y),
            aim_point,
            (255, 0, 255),
            1,
        )

    mode_text = (
        "1/3"
        if target_mode == "enemy"
        else "0/2"
    )

    information = [
        f"ROI: {roi_size}x{roi_size}",
        f"FPS: {fps:.1f}",
        f"Target classes: {mode_text}",
        f"X strength: {x_strength:.2f}",
        f"Y strength: {y_strength:.2f}",
        f"Smooth: {smoothing:.2f}",
        f"Aim region: {aim_region:.2f}",
        f"RMB aim: {'ON' if aiming else 'OFF'}",
        "F1:1/3 F2:0/2",
        "F3/F4:X  F5/F6:Y  F7/F8:Smooth",
        "[/]:Aim region  F9/F10:ROI",
        "F11:Debug  END:Quit",
    ]

    for index, text in enumerate(information):
        if index == 7:
            color = (
                (0, 255, 0)
                if aiming
                else (170, 170, 170)
            )
        else:
            color = (255, 255, 255)

        cv2.putText(
            frame,
            text,
            (10, 22 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )


# ============================================================
# 主程序
# ============================================================

def main():
    enable_dpi_awareness()
    args = parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到模型：{model_path}"
        )

    model = YOLO(str(model_path))

    roi_size = max(100, args.roi_size)

    x_strength = max(0.0, args.x_strength)
    y_strength = max(0.0, args.y_strength)

    smoothing = clamp(
        args.smooth,
        0.0,
        0.95,
    )

    aim_region = clamp(
        args.aim_region,
        0.0,
        1.0,
    )

    # 默认锁定1/3
    target_mode = "enemy"
    target_classes = {1, 3}

    debug_enabled = not args.no_debug
    debug_window_created = False

    hotkeys = HotkeyState()
    aim_controller = AimController()

    window_name = "YOLO Aim Debug"

    previous_time = time.perf_counter()
    fps = 0.0

    print(f"模型：{model_path}")
    print("F1：锁定T")
    print("F2：锁定CT")
    print("F3/F4：降低/提高X轴强度")
    print("F5/F6：降低/提高Y轴强度")
    print("F7/F8：降低/提高平滑强度")
    print("[ / ]：降低/提高锁点区域")
    print("F9/F10：缩小/扩大检测区域")
    print("F11：显示/关闭调试窗口")
    print("按住右键：建立并保持单目标锁定")
    print("END：退出")

    with mss.mss() as capture:
        if not (
            1 <= args.monitor < len(capture.monitors)
        ):
            raise ValueError(
                f"显示器编号{args.monitor}不存在，"
                f"可用范围为1～"
                f"{len(capture.monitors) - 1}"
            )

        monitor = capture.monitors[args.monitor]

        print(
            f"显示器：{monitor['width']}×"
            f"{monitor['height']}"
        )

        running = True

        while running:
            # ------------------------------------------------
            # 全局快捷键
            # ------------------------------------------------

            if hotkeys.pressed_once(VK_END):
                break

            if hotkeys.pressed_once(VK_F1):
                target_mode = "enemy"
                target_classes = {1, 3}

                aim_controller.reset()

                print("锁定类别：1/3")

            if hotkeys.pressed_once(VK_F2):
                target_mode = "friendly"
                target_classes = {0, 2}

                aim_controller.reset()

                print("锁定类别：0/2")

            if hotkeys.pressed_once(VK_F3):
                x_strength = max(
                    0.0,
                    x_strength - 0.05,
                )

                print(
                    f"X轴强度：{x_strength:.2f}"
                )

            if hotkeys.pressed_once(VK_F4):
                x_strength = min(
                    3.0,
                    x_strength + 0.05,
                )

                print(
                    f"X轴强度：{x_strength:.2f}"
                )

            if hotkeys.pressed_once(VK_F5):
                y_strength = max(
                    0.0,
                    y_strength - 0.05,
                )

                print(
                    f"Y轴强度：{y_strength:.2f}"
                )

            if hotkeys.pressed_once(VK_F6):
                y_strength = min(
                    3.0,
                    y_strength + 0.05,
                )

                print(
                    f"Y轴强度：{y_strength:.2f}"
                )

            if hotkeys.pressed_once(VK_F7):
                smoothing = max(
                    0.0,
                    smoothing - 0.05,
                )

                aim_controller.reset()

                print(
                    f"平滑强度：{smoothing:.2f}"
                )

            if hotkeys.pressed_once(VK_F8):
                smoothing = min(
                    0.95,
                    smoothing + 0.05,
                )

                aim_controller.reset()

                print(
                    f"平滑强度：{smoothing:.2f}"
                )

            if hotkeys.pressed_once(VK_OEM_4):
                aim_region = max(
                    0.0,
                    aim_region - 0.05,
                )

                aim_region = round(
                    aim_region,
                    2,
                )

                aim_controller.reset()

                print(
                    f"锁点区域：{aim_region:.2f}"
                )

            if hotkeys.pressed_once(VK_OEM_6):
                aim_region = min(
                    1.0,
                    aim_region + 0.05,
                )

                aim_region = round(
                    aim_region,
                    2,
                )

                aim_controller.reset()

                print(
                    f"锁点区域：{aim_region:.2f}"
                )

            if hotkeys.pressed_once(VK_F9):
                roi_size = max(
                    100,
                    roi_size - 50,
                )

                aim_controller.reset()

                print(
                    f"检测区域：{roi_size}×{roi_size}"
                )

            if hotkeys.pressed_once(VK_F10):
                roi_size = min(
                    monitor["width"],
                    monitor["height"],
                    roi_size + 50,
                )

                aim_controller.reset()

                print(
                    f"检测区域：{roi_size}×{roi_size}"
                )

            if hotkeys.pressed_once(VK_F11):
                debug_enabled = not debug_enabled

                print(
                    "调试窗口："
                    + (
                        "开启"
                        if debug_enabled
                        else "关闭"
                    )
                )

                if (
                    not debug_enabled
                    and debug_window_created
                ):
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        pass

                    debug_window_created = False

            # ------------------------------------------------
            # 截取屏幕中心
            # ------------------------------------------------

            roi = calculate_center_roi(
                monitor,
                roi_size,
            )

            screenshot = capture.grab(roi)

            frame = np.asarray(screenshot)

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR,
            )

            # ------------------------------------------------
            # YOLO连续追踪
            # ------------------------------------------------

            track_arguments = {
                "source": frame,
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "classes": [0, 1, 2, 3],
                "tracker": "bytetrack.yaml",
                "persist": True,
                "verbose": False,
            }

            if args.device is not None:
                track_arguments["device"] = args.device

            results = model.track(
                **track_arguments
            )

            detections = extract_detections(
                results[0]
            )

            frame_height, frame_width = frame.shape[:2]

            center_x = frame_width / 2.0
            center_y = frame_height / 2.0

            # ------------------------------------------------
            # 锁敌逻辑
            # ------------------------------------------------

            target = select_target(
                detections=detections,
                allowed_classes=target_classes,
                center_x=center_x,
                center_y=center_y,
                aim_region=aim_region,
            )

            aiming = is_key_down(VK_RBUTTON)

            # ------------------------------------------------
            # 计算并发送鼠标移动
            # ------------------------------------------------

            if aiming and target is not None:
                move_x, move_y, _, _ = calculate_mouse_movement(
                    target=target,
                    detections=detections,
                    center_x=center_x,
                    center_y=center_y,
                    aim_region=aim_region,
                    x_strength=x_strength,
                    y_strength=y_strength,
                    max_step=args.max_step,
                    deadzone=args.deadzone,
                    invert_y=args.invert_y,
                )

                aim_controller.move(
                    move_x,
                    move_y,
                    smoothing,
                )

            else:
                aim_controller.reset()

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            if elapsed > 0:
                current_fps = 1.0 / elapsed

                fps = (
                    current_fps
                    if fps == 0
                    else fps * 0.90
                    + current_fps * 0.10
                )

            # ------------------------------------------------
            # 调试窗口
            # ------------------------------------------------

            if debug_enabled:
                if not debug_window_created:
                    cv2.namedWindow(
                        window_name,
                        cv2.WINDOW_NORMAL,
                    )

                    debug_window_created = True

                debug_frame = frame.copy()

                draw_debug(
                    frame=debug_frame,
                    detections=detections,
                    target=target,
                    fps=fps,
                    roi_size=roi_size,
                    target_mode=target_mode,
                    x_strength=x_strength,
                    y_strength=y_strength,
                    smoothing=smoothing,
                    aim_region=aim_region,
                    aiming=aiming,
                )

                cv2.imshow(
                    window_name,
                    debug_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    running = False

            else:
                cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()