import argparse
import ctypes
import time
from pathlib import Path

import cv2
import mss
import numpy as np
from ultralytics import YOLO


# 工程根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 默认模型位置
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "enemy_detector"
    / "weights"
    / "best.pt"
)

# 四个类别
CLASS_NAMES = {
    0: "fb",
    1: "eb",
    2: "fh",
    3: "eh",
}

# OpenCV 使用 BGR 颜色
CLASS_COLORS = {
    0: (0, 255, 0),      # 好人躯干：绿色
    1: (0, 0, 255),      # 敌人躯干：红色
    2: (255, 255, 0),    # 好人头部：青色
    3: (0, 165, 255),    # 敌人头部：橙色
}


def enable_windows_dpi_awareness():
    """
    避免 Windows 设置了 125%、150% 显示缩放时，
    截屏坐标与实际屏幕坐标不一致。
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="检测屏幕中心区域内的 YOLO 目标"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="YOLO 模型权重路径",
    )

    parser.add_argument(
        "--roi-size",
        type=int,
        default=400,
        help="屏幕中心识别区域大小，默认 400×400",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="按加减键时，每次调整的像素数",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="置信度阈值",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS 的 IoU 阈值",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO 推理尺寸",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备，例如 0、cpu；不填写则自动选择",
    )

    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="显示器编号，主显示器通常为 1",
    )

    return parser.parse_args()


def calculate_center_roi(monitor, roi_size):
    """
    根据显示器尺寸计算屏幕中心正方形区域。
    """
    roi_size = min(
        roi_size,
        monitor["width"],
        monitor["height"],
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
        "width": int(roi_size),
        "height": int(roi_size),
    }


def draw_detections(frame, result):
    """
    将 YOLO 检测框、类别、置信度和目标中心画到图像上。
    """
    if result.boxes is None:
        return frame, 0

    box_data = result.boxes.data.cpu().numpy()
    detection_count = 0

    for detection in box_data:
        x1, y1, x2, y2, confidence, class_id = detection[:6]

        class_id = int(class_id)

        # 只显示 0、1、2、3
        if class_id not in CLASS_NAMES:
            continue

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        color = CLASS_COLORS[class_id]
        class_name = CLASS_NAMES[class_id]

        # 绘制检测框
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # 检测框中心
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        cv2.circle(
            frame,
            (center_x, center_y),
            4,
            color,
            -1,
        )

        label = f"{class_id} {class_name} {confidence:.2f}"

        # 计算文字尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        label_top = max(y1, text_height + 8)

        # 标签背景
        cv2.rectangle(
            frame,
            (x1, label_top - text_height - 8),
            (x1 + text_width + 6, label_top),
            color,
            -1,
        )

        # 标签文字
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        detection_count += 1

    return frame, detection_count


def draw_interface(frame, fps, roi_size, detection_count):
    """
    绘制中心准星、FPS 和快捷键说明。
    """
    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    # 屏幕中心准星
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

    information = [
        f"ROI: {roi_size}x{roi_size}",
        f"FPS: {fps:.1f}",
        f"Objects: {detection_count}",
        "[+] enlarge  [-] reduce  [Q/ESC] quit",
    ]

    for index, text in enumerate(information):
        cv2.putText(
            frame,
            text,
            (10, 25 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    enable_windows_dpi_awareness()
    args = parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"没有找到模型权重：{model_path}\n"
            "请确认 best.pt 位于：\n"
            "runs/enemy_detector/weights/best.pt"
        )

    if args.roi_size < 100:
        raise ValueError("识别区域不能小于 100×100")

    print(f"加载模型：{model_path}")
    model = YOLO(str(model_path))

    roi_size = args.roi_size
    window_name = "YOLO Screen Detection"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with mss.mss() as screen_capture:
        if args.monitor >= len(screen_capture.monitors):
            raise ValueError(
                f"显示器编号 {args.monitor} 不存在。"
                f"当前可用编号为 1～{len(screen_capture.monitors) - 1}"
            )

        monitor = screen_capture.monitors[args.monitor]

        print(
            f"显示器：{args.monitor}，"
            f"分辨率：{monitor['width']}×{monitor['height']}"
        )
        print(f"初始识别区域：{roi_size}×{roi_size}")
        print("快捷键：+ 扩大，- 缩小，Q 或 Esc 退出")

        previous_time = time.perf_counter()
        fps = 0.0

        while True:
            roi = calculate_center_roi(monitor, roi_size)

            # 截取屏幕中心区域
            screenshot = screen_capture.grab(roi)

            # MSS 返回 BGRA，转换成 OpenCV BGR
            frame = np.asarray(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            predict_arguments = {
                "source": frame,
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "classes": [0, 1, 2, 3],
                "verbose": False,
            }

            if args.device is not None:
                predict_arguments["device"] = args.device

            results = model.predict(**predict_arguments)
            result = results[0]

            frame, detection_count = draw_detections(
                frame,
                result,
            )

            current_time = time.perf_counter()
            frame_time = current_time - previous_time
            previous_time = current_time

            if frame_time > 0:
                current_fps = 1.0 / frame_time
                fps = (
                    current_fps
                    if fps == 0
                    else fps * 0.9 + current_fps * 0.1
                )

            draw_interface(
                frame,
                fps,
                roi_size,
                detection_count,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            # Q 或 Esc 退出
            if key in (ord("q"), ord("Q"), 27):
                break

            # + 或 = 扩大区域
            elif key in (ord("+"), ord("=")):
                roi_size += args.step
                roi_size = min(
                    roi_size,
                    monitor["width"],
                    monitor["height"],
                )
                print(f"识别区域：{roi_size}×{roi_size}")

            # - 或 _ 缩小区域
            elif key in (ord("-"), ord("_")):
                roi_size = max(
                    100,
                    roi_size - args.step,
                )
                print(f"识别区域：{roi_size}×{roi_size}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()