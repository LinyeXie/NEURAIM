import argparse
import ctypes
import time
from pathlib import Path

import cv2
import mss
import numpy as np
from ultralytics import YOLO


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
    0: (0, 255, 0),       # 绿色：友方躯干
    1: (0, 0, 255),       # 红色：敌方躯干
    2: (255, 255, 0),     # 青色：友方头部
    3: (0, 165, 255),     # 橙色：敌方头部
}

# Windows 虚拟按键
VK_RBUTTON = 0x02
VK_END = 0x23

VK_F1 = 0x70
VK_F2 = 0x71
VK_F3 = 0x72
VK_F4 = 0x73
VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79

MOUSEEVENTF_MOVE = 0x0001

user32 = ctypes.windll.user32


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


def enable_dpi_awareness():
    """避免 Windows 显示缩放造成截屏坐标偏移。"""
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
    """检测全局热键的单次按下事件。"""

    def __init__(self):
        self.previous = {}

    def pressed_once(self, vk_code):
        current = is_key_down(vk_code)
        previous = self.previous.get(vk_code, False)
        self.previous[vk_code] = current
        return current and not previous


class AimController:
    """发送相对鼠标移动，并保留小数移动余量。"""

    def __init__(self):
        self.residual_x = 0.0
        self.residual_y = 0.0

    def reset(self):
        self.residual_x = 0.0
        self.residual_y = 0.0

    def move(self, dx, dy):
        self.residual_x += dx
        self.residual_y += dy

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO 屏幕检测及离线游戏自动瞄准测试"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="YOLO 权重路径",
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
        help="NMS IoU 阈值",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="YOLO 推理尺寸",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备，例如 0 或 cpu",
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
        help="X 轴瞄准强度",
    )

    parser.add_argument(
        "--y-strength",
        type=float,
        default=0.30,
        help="Y 轴瞄准强度",
    )

    parser.add_argument(
        "--max-step",
        type=float,
        default=80.0,
        help="单帧最大鼠标移动量",
    )

    parser.add_argument(
        "--deadzone",
        type=float,
        default=3.0,
        help="准星中心死区，防止抖动",
    )

    parser.add_argument(
        "--body-y",
        type=float,
        default=0.32,
        help="躯干瞄准点高度，0为框顶部，1为框底部",
    )

    parser.add_argument(
        "--invert-y",
        action="store_true",
        help="反转 Y 轴移动",
    )

    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="启动时关闭调试窗口",
    )

    return parser.parse_args()


def calculate_center_roi(monitor, roi_size):
    roi_size = max(
        100,
        min(roi_size, monitor["width"], monitor["height"]),
    )

    left = monitor["left"] + (monitor["width"] - roi_size) // 2
    top = monitor["top"] + (monitor["height"] - roi_size) // 2

    return {
        "left": int(left),
        "top": int(top),
        "width": int(roi_size),
        "height": int(roi_size),
    }


def extract_detections(result):
    detections = []

    if result.boxes is None:
        return detections

    for detection in result.boxes.data.cpu().numpy():
        x1, y1, x2, y2, confidence, class_id = detection[:6]

        class_id = int(class_id)

        if class_id not in CLASS_NAMES:
            continue

        detections.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(confidence),
            "class_id": class_id,
        })

    return detections


def get_aim_point(detection, body_y):
    x1 = detection["x1"]
    y1 = detection["y1"]
    x2 = detection["x2"]
    y2 = detection["y2"]
    class_id = detection["class_id"]

    aim_x = (x1 + x2) / 2.0

    if class_id == 3:
        # 敌方头部：瞄准检测框中心
        aim_y = (y1 + y2) / 2.0
    else:
        # 敌方躯干：瞄准上胸位置
        aim_y = y1 + (y2 - y1) * body_y

    return aim_x, aim_y


def select_target(detections, center_x, center_y, body_y):
    """
    优先选择敌方头部 class 3。
    没有头部时选择敌方躯干 class 1。
    同类别中选择距离准星最近的目标。
    """
    enemy_heads = [
        detection
        for detection in detections
        if detection["class_id"] == 3
    ]

    enemy_bodies = [
        detection
        for detection in detections
        if detection["class_id"] == 1
    ]

    candidates = enemy_heads if enemy_heads else enemy_bodies

    if not candidates:
        return None

    def target_distance(detection):
        aim_x, aim_y = get_aim_point(detection, body_y)

        return (
            (aim_x - center_x) ** 2
            + (aim_y - center_y) ** 2
        )

    return min(candidates, key=target_distance)


def calculate_mouse_movement(
    target,
    center_x,
    center_y,
    x_strength,
    y_strength,
    max_step,
    deadzone,
    body_y,
    invert_y,
):
    aim_x, aim_y = get_aim_point(target, body_y)

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

    move_x = max(-max_step, min(max_step, move_x))
    move_y = max(-max_step, min(max_step, move_y))

    return move_x, move_y, aim_x, aim_y


def draw_debug(
    frame,
    detections,
    target,
    fps,
    roi_size,
    x_strength,
    y_strength,
    aiming,
    body_y,
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

        thickness = 3 if detection is target else 2

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        label = (
            f"{class_id} {CLASS_NAMES[class_id]} "
            f"{detection['confidence']:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    # 准星中心
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

    if target is not None:
        aim_x, aim_y = get_aim_point(target, body_y)
        aim_point = (int(aim_x), int(aim_y))

        cv2.circle(
            frame,
            aim_point,
            6,
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

    status_color = (0, 255, 0) if aiming else (180, 180, 180)

    information = [
        f"ROI: {roi_size}x{roi_size}",
        f"FPS: {fps:.1f}",
        f"X strength: {x_strength:.2f}",
        f"Y strength: {y_strength:.2f}",
        f"RMB aim: {'ON' if aiming else 'OFF'}",
        "F1/F2 X- X+ | F3/F4 Y- Y+",
        "F8 debug | F9/F10 ROI- ROI+ | END quit",
    ]

    for index, text in enumerate(information):
        color = status_color if index == 4 else (255, 255, 255)

        cv2.putText(
            frame,
            text,
            (10, 24 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


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

    debug_enabled = not args.no_debug
    debug_window_created = False

    hotkeys = HotkeyState()
    aim_controller = AimController()

    window_name = "YOLO Aim Debug"

    previous_time = time.perf_counter()
    fps = 0.0

    print(f"模型：{model_path}")
    print("按住鼠标右键：自动瞄准")
    print("F1/F2：降低/提高 X 轴强度")
    print("F3/F4：降低/提高 Y 轴强度")
    print("F8：显示或关闭调试窗口")
    print("F9/F10：缩小或扩大检测范围")
    print("END：退出程序")

    with mss.mss() as capture:
        if args.monitor >= len(capture.monitors):
            raise ValueError(
                f"显示器 {args.monitor} 不存在，可用范围为 "
                f"1～{len(capture.monitors) - 1}"
            )

        monitor = capture.monitors[args.monitor]

        print(
            f"检测显示器：{monitor['width']}×{monitor['height']}"
        )

        running = True

        while running:
            # 全局热键
            if hotkeys.pressed_once(VK_END):
                break

            if hotkeys.pressed_once(VK_F1):
                x_strength = max(0.0, x_strength - 0.05)
                print(f"X 轴强度：{x_strength:.2f}")

            if hotkeys.pressed_once(VK_F2):
                x_strength = min(3.0, x_strength + 0.05)
                print(f"X 轴强度：{x_strength:.2f}")

            if hotkeys.pressed_once(VK_F3):
                y_strength = max(0.0, y_strength - 0.05)
                print(f"Y 轴强度：{y_strength:.2f}")

            if hotkeys.pressed_once(VK_F4):
                y_strength = min(3.0, y_strength + 0.05)
                print(f"Y 轴强度：{y_strength:.2f}")

            if hotkeys.pressed_once(VK_F8):
                debug_enabled = not debug_enabled
                print(
                    "调试窗口："
                    + ("开启" if debug_enabled else "关闭")
                )

                if not debug_enabled and debug_window_created:
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        pass

                    debug_window_created = False

            if hotkeys.pressed_once(VK_F9):
                roi_size = max(100, roi_size - 50)
                print(f"检测范围：{roi_size}×{roi_size}")

            if hotkeys.pressed_once(VK_F10):
                roi_size = min(
                    monitor["width"],
                    monitor["height"],
                    roi_size + 50,
                )
                print(f"检测范围：{roi_size}×{roi_size}")

            roi = calculate_center_roi(monitor, roi_size)

            screenshot = capture.grab(roi)
            frame = np.asarray(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=[0, 1, 2, 3],
                device=args.device,
                verbose=False,
            )

            detections = extract_detections(results[0])

            frame_height, frame_width = frame.shape[:2]
            center_x = frame_width / 2.0
            center_y = frame_height / 2.0

            target = select_target(
                detections,
                center_x,
                center_y,
                args.body_y,
            )

            aiming = is_key_down(VK_RBUTTON)

            if aiming and target is not None:
                (
                    move_x,
                    move_y,
                    _,
                    _,
                ) = calculate_mouse_movement(
                    target=target,
                    center_x=center_x,
                    center_y=center_y,
                    x_strength=x_strength,
                    y_strength=y_strength,
                    max_step=args.max_step,
                    deadzone=args.deadzone,
                    body_y=args.body_y,
                    invert_y=args.invert_y,
                )

                aim_controller.move(move_x, move_y)
            else:
                aim_controller.reset()

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = (
                    current_fps
                    if fps == 0
                    else fps * 0.9 + current_fps * 0.1
                )

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
                    x_strength=x_strength,
                    y_strength=y_strength,
                    aiming=aiming,
                    body_y=args.body_y,
                )

                cv2.imshow(window_name, debug_frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    running = False
            else:
                # 保持 OpenCV 事件处理，但不显示窗口
                cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()