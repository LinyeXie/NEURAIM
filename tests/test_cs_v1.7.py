import argparse
import atexit
import ctypes
import threading
import time
from ctypes import wintypes
from pathlib import Path

import cv2
import mss
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ops as yolo_ops

try:
    # 新版 Ultralytics
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    # 兼容仍将 NMS 放在 ops 中的旧版 Ultralytics
    from ultralytics.utils.ops import non_max_suppression


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

DEFAULT_CAPTURE_FPS = 90.0
DEFAULT_DEBUG_FPS = 30.0
DEFAULT_MAX_DET = 32
WINDOW_MESSAGE_FPS = 60.0


# ============================================================
# Windows 按键
# ============================================================

VK_RBUTTON = 0x02

VK_END = 0x23
VK_HOME = 0x24

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
# 独立参数悬浮面板
# ============================================================

class ParameterOverlay:
    """
    独立于调试窗口的左上角参数面板。

    面板无边框、背景透明、鼠标穿透，并持续保持置顶。
    """

    PANEL_WIDTH = 430
    PANEL_HEIGHT = 216

    # 参数显示位置（屏幕左上角为0, 0）。
    # 增大X向右移动，增大Y向下移动。
    POSITION_X = 100
    POSITION_Y = 100

    GWL_STYLE = -16
    GWL_EXSTYLE = -20

    WS_CAPTION = 0x00C00000
    WS_THICKFRAME = 0x00040000
    WS_MINIMIZEBOX = 0x00020000
    WS_MAXIMIZEBOX = 0x00010000
    WS_SYSMENU = 0x00080000

    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_NOACTIVATE = 0x08000000
    WS_EX_TOPMOST = 0x00000008

    LWA_COLORKEY = 0x00000001
    TRANSPARENT_COLORREF = 0x00FF00FF

    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_NOACTIVATE = 0x0010
    SWP_FRAMECHANGED = 0x0020
    SWP_SHOWWINDOW = 0x0040
    SWP_NOOWNERZORDER = 0x0200

    GA_ROOT = 2
    TOPMOST_REFRESH_INTERVAL = 0.20

    def __init__(self):
        self.window_name = "YOLO Parameters"
        self.created = False
        self.hwnd = None
        self.last_topmost_refresh = 0.0
        self.last_content = None
        self._configure_win32_functions()

    @staticmethod
    def _configure_win32_functions():
        """
        显式声明64位Windows下的句柄和参数类型。

        ctypes未声明签名时会把部分值按32位int处理，可能导致
        SetWindowPos实际收到无效窗口句柄，但程序表面上不报错。
        """
        user32.FindWindowW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
        ]
        user32.FindWindowW.restype = wintypes.HWND

        user32.GetAncestor.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
        ]
        user32.GetAncestor.restype = wintypes.HWND

        user32.GetWindowLongW.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
        ]
        user32.GetWindowLongW.restype = ctypes.c_long

        user32.SetWindowLongW.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_long,
        ]
        user32.SetWindowLongW.restype = ctypes.c_long

        user32.SetLayeredWindowAttributes.argtypes = [
            wintypes.HWND,
            wintypes.COLORREF,
            wintypes.BYTE,
            wintypes.DWORD,
        ]
        user32.SetLayeredWindowAttributes.restype = wintypes.BOOL

        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL

    @staticmethod
    def put_high_contrast_text(
        image,
        text,
        position,
        color,
        scale=0.52,
    ):
        # 先绘制3像素深色底字，再绘制2像素亮色正文。
        # 两层只相差1像素，形成清晰但不过厚的文字描边。
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (100, 0, 0),
            3,
            cv2.LINE_8,
        )
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_8,
        )

    def configure_window(self):
        hwnd = user32.FindWindowW(
            None,
            self.window_name,
        )

        if not hwnd:
            return

        # 某些OpenCV后端返回的是内部窗口，样式与Z序必须作用于
        # 它所属的真实顶层窗口。
        root_hwnd = user32.GetAncestor(
            hwnd,
            self.GA_ROOT,
        )
        self.hwnd = root_hwnd or hwnd
        hwnd = self.hwnd

        style = user32.GetWindowLongW(
            hwnd,
            self.GWL_STYLE,
        )
        style &= ~(
            self.WS_CAPTION
            | self.WS_THICKFRAME
            | self.WS_MINIMIZEBOX
            | self.WS_MAXIMIZEBOX
            | self.WS_SYSMENU
        )
        user32.SetWindowLongW(
            hwnd,
            self.GWL_STYLE,
            style,
        )

        extended_style = user32.GetWindowLongW(
            hwnd,
            self.GWL_EXSTYLE,
        )
        extended_style |= (
            self.WS_EX_LAYERED
            | self.WS_EX_TRANSPARENT
            | self.WS_EX_TOOLWINDOW
            | self.WS_EX_NOACTIVATE
            | self.WS_EX_TOPMOST
        )
        user32.SetWindowLongW(
            hwnd,
            self.GWL_EXSTYLE,
            extended_style,
        )

        # 洋红色作为透明色：背景完全透明，文字保持不透明。
        user32.SetLayeredWindowAttributes(
            hwnd,
            self.TRANSPARENT_COLORREF,
            255,
            self.LWA_COLORKEY,
        )

        # 移除标题栏后必须刷新非客户区，并把窗口外框强制调整为
        # 图像的准确尺寸，否则原标题栏/边框占用的位置会变成黑条。
        user32.SetWindowPos(
            hwnd,
            self.HWND_TOPMOST,
            self.POSITION_X,
            self.POSITION_Y,
            self.PANEL_WIDTH,
            self.PANEL_HEIGHT,
            (
                self.SWP_NOACTIVATE
                | self.SWP_FRAMECHANGED
                | self.SWP_SHOWWINDOW
            ),
        )
        self.last_topmost_refresh = 0.0

    def keep_topmost(self, force=False):
        if not self.hwnd:
            self.configure_window()

        if not self.hwnd:
            return

        now = time.perf_counter()

        # 窗口已带WS_EX_TOPMOST，不必在每个识别帧重复调用Win32。
        # 定期重新插入topmost层即可维持层级，减少高帧率循环开销。
        if (
            not force
            and now - self.last_topmost_refresh
            < self.TOPMOST_REFRESH_INTERVAL
        ):
            return

        flags = (
            self.SWP_NOMOVE
            | self.SWP_NOSIZE
            | self.SWP_NOACTIVATE
            | self.SWP_NOOWNERZORDER
        )

        user32.SetWindowPos(
            self.hwnd,
            self.HWND_NOTOPMOST,
            0,
            0,
            0,
            0,
            flags,
        )

        user32.SetWindowPos(
            self.hwnd,
            self.HWND_TOPMOST,
            0,
            0,
            0,
            0,
            (
                self.SWP_NOMOVE
                | self.SWP_NOSIZE
                | self.SWP_NOACTIVATE
                | self.SWP_NOOWNERZORDER
                | self.SWP_SHOWWINDOW
            ),
        )
        self.last_topmost_refresh = now

    def show(
        self,
        target_mode,
        x_strength,
        y_strength,
        smoothing,
        aim_region,
        roi_size,
        aiming,
    ):
        # 参数没有变化时不重复生成图像和调用imshow。悬浮层的置顶
        # 仍由主循环单独维护，因此不会影响窗口层级。
        content = (
            target_mode,
            round(x_strength, 2),
            round(y_strength, 2),
            round(smoothing, 2),
            round(aim_region, 2),
            int(roi_size),
            bool(aiming),
        )

        if self.created and content == self.last_content:
            return

        panel = np.full(
            (
                self.PANEL_HEIGHT,
                self.PANEL_WIDTH,
                3,
            ),
            (255, 0, 255),
            dtype=np.uint8,
        )

        target_text = (
            "Enemy (1/3)"
            if target_mode == "enemy"
            else "Friendly (0/2)"
        )

        lines = [
            (
                "PARAMETERS",
                (0, 255, 255),
            ),
            (
                f"Target [F1/F2] : {target_text}",
                (255, 255, 255),
            ),
            (
                f"X strength [F3/F4] : {x_strength:.2f}",
                (255, 255, 255),
            ),
            (
                f"Y strength [F5/F6] : {y_strength:.2f}",
                (255, 255, 255),
            ),
            (
                f"Smooth [F7/F8] : {smoothing:.2f}",
                (255, 255, 255),
            ),
            (
                f"Aim point : Body <- [ / ] -> Head  "
                f"{aim_region:.2f}",
                (255, 255, 255),
            ),
            (
                f"Detection [F9/F10] : "
                f"{roi_size}x{roi_size}",
                (255, 255, 255),
            ),
            (
                f"RMB aim : {'ACTIVE' if aiming else 'READY'}",
                (
                    (0, 255, 0)
                    if aiming
                    else (0, 220, 255)
                ),
            ),
        ]

        for index, (text, color) in enumerate(lines):
            self.put_high_contrast_text(
                panel,
                text,
                (10, 24 + index * 25),
                color,
            )

        if not self.created:
            cv2.namedWindow(
                self.window_name,
                cv2.WINDOW_NORMAL,
            )
            cv2.resizeWindow(
                self.window_name,
                self.PANEL_WIDTH,
                self.PANEL_HEIGHT,
            )
            self.created = True

        cv2.imshow(
            self.window_name,
            panel,
        )

        self.last_content = content
        self.keep_topmost()

    def close(self):
        if not self.created:
            return

        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

        self.created = False
        self.hwnd = None
        self.last_content = None


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


def enable_high_resolution_timer():
    """
    将当前进程的Windows计时器粒度请求为1 ms。

    这不会创建进程或线程，只减少限帧等待的过度休眠。注册退出回调
    可确保程序结束时释放计时器请求。
    """
    try:
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod.argtypes = [wintypes.UINT]
        winmm.timeBeginPeriod.restype = wintypes.UINT
        winmm.timeEndPeriod.argtypes = [wintypes.UINT]
        winmm.timeEndPeriod.restype = wintypes.UINT

        if winmm.timeBeginPeriod(1) == 0:
            atexit.register(winmm.timeEndPeriod, 1)
    except Exception:
        pass


def is_key_down(vk_code):
    return bool(user32.GetAsyncKeyState(vk_code) & 0x8000)


def poll_window_key():
    """处理OpenCV窗口消息，但不主动等待下一帧。"""

    poll_key = getattr(cv2, "pollKey", None)

    if poll_key is not None:
        return poll_key() & 0xFF

    # 兼容没有pollKey的旧版OpenCV。
    return cv2.waitKey(1) & 0xFF


def wait_until(deadline):
    """
    精确等待到目标时刻。

    Windows上的普通sleep可能明显超时。先睡到截止时间前约1毫秒，
    最后一小段使用让出线程的短等待，减少限帧器因过度休眠而低于
    设定帧率的情况。
    """
    while True:
        remaining = deadline - time.perf_counter()

        if remaining <= 0:
            return

        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        else:
            time.sleep(0)


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

    @staticmethod
    def _smooth_axis(
        target,
        smoothed,
        residual,
        alpha,
    ):
        """
        对单轴移动量进行无过冲平滑。

        普通指数平滑会保留上一帧的方向和速度；接近锁点时，
        即使当前指令已经减小或反向，旧移动量仍可能把准星推过
        目标。这里在停止或反向时立即消除历史惯性，并限制平滑
        输出不能大于当前帧指令。
        """
        if target == 0.0:
            return 0.0, 0.0

        # 当前指令与历史输出方向相反，说明已经越过锁点。
        # 本帧先制动并清除小数余量，下一帧再按新误差修正。
        if smoothed * target < 0.0:
            return 0.0, 0.0

        if smoothed == 0.0:
            smoothed = target
        else:
            smoothed += (
                target - smoothed
            ) * alpha

        # 目标越近，当前指令越小；禁止历史平滑量大于当前指令，
        # 避免接近锁点后仍按旧速度继续前冲。
        current_limit = abs(target)
        smoothed = clamp(
            smoothed,
            -current_limit,
            current_limit,
        )

        # 小数余量也不能带着相反方向跨过锁点。
        if residual * target < 0.0:
            residual = 0.0

        return smoothed, residual

    def move(self, target_dx, target_dy, smoothing):
        smoothing = clamp(smoothing, 0.0, 0.95)

        # 平滑越高，新输入所占比例越低
        alpha = max(0.05, 1.0 - smoothing)

        if not self.initialized:
            self.smoothed_x = target_dx
            self.smoothed_y = target_dy
            self.initialized = True
        else:
            (
                self.smoothed_x,
                self.residual_x,
            ) = self._smooth_axis(
                target_dx,
                self.smoothed_x,
                self.residual_x,
                alpha,
            )

            (
                self.smoothed_y,
                self.residual_y,
            ) = self._smooth_axis(
                target_dy,
                self.smoothed_y,
                self.residual_y,
                alpha,
            )

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
        default=300,
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
        default=192,
        help="YOLO推理尺寸",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备，例如0或cpu；默认自动选择",
    )

    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="显示器编号",
    )

    parser.add_argument(
        "--capture-fps",
        type=float,
        default=DEFAULT_CAPTURE_FPS,
        help="采集与识别循环帧率上限，默认90 FPS",
    )

    parser.add_argument(
        "--debug-fps",
        type=float,
        default=DEFAULT_DEBUG_FPS,
        help="调试窗口刷新率；不限制识别与控制帧率，默认30 FPS",
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=DEFAULT_MAX_DET,
        help="每帧NMS后最多保留的检测框数量，默认32",
    )

    parser.add_argument(
        "--inference-backend",
        choices=("direct", "predict"),
        default="direct",
        help=(
            "推理后端：direct绕过Ultralytics Predictor以降低每帧开销；"
            "不兼容时自动回退predict"
        ),
    )

    parser.add_argument(
        "--x-strength",
        type=float,
        default=0.6,
        help="X轴移动强度",
    )

    parser.add_argument(
        "--y-strength",
        type=float,
        default=0.6,
        help="Y轴移动强度",
    )

    parser.add_argument(
        "--smooth",
        type=float,
        default=0.30,
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
        default=100.0,
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


class PipelinedScreenCapture:
    """
    只保留最新一帧的双缓冲截屏器。

    截图线程在主线程处理当前帧时采集下一帧，使mss.grab与GPU推理
    尽量重叠。它只创建一个后台线程，不会创建额外进程，也不会累积
    旧帧而增加锁定延迟。
    """

    def __init__(self, monitor, roi_size):
        self.monitor = dict(monitor)
        self.roi_size = int(roi_size)

        self.condition = threading.Condition()
        self.request = threading.Event()
        self.stopped = threading.Event()

        self.frame = None
        self.frame_generation = -1
        self.generation = 0
        self.error = None

        self.thread = threading.Thread(
            target=self._capture_loop,
            name="ScreenCapture",
            daemon=True,
        )
        self.thread.start()
        self.request.set()

    def _capture_loop(self):
        try:
            with mss.mss() as capture:
                while not self.stopped.is_set():
                    if not self.request.wait(0.1):
                        continue

                    self.request.clear()

                    if self.stopped.is_set():
                        break

                    with self.condition:
                        roi_size = self.roi_size
                        generation = self.generation

                    roi = calculate_center_roi(
                        self.monitor,
                        roi_size,
                    )
                    screenshot = capture.grab(roi)

                    # MSS返回BGRA。这里只进行一次连续化复制，同时去掉
                    # alpha通道，得到YOLO需要的BGR输入。
                    frame = np.ascontiguousarray(
                        np.asarray(screenshot)[:, :, :3]
                    )

                    with self.condition:
                        # 检测区域在采集期间发生变化时丢弃旧尺寸帧。
                        if generation == self.generation:
                            self.frame = frame
                            self.frame_generation = generation

                        self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def set_roi_size(self, roi_size):
        roi_size = int(roi_size)

        with self.condition:
            if roi_size == self.roi_size:
                return

            self.roi_size = roi_size
            self.generation += 1
            self.frame = None
            self.frame_generation = -1

        self.request.set()

    def get_frame(self, timeout=1.0):
        deadline = time.perf_counter() + timeout

        with self.condition:
            while (
                self.frame is None
                or self.frame_generation != self.generation
            ):
                if self.error is not None:
                    raise RuntimeError(
                        "屏幕采集线程异常"
                    ) from self.error

                remaining = deadline - time.perf_counter()

                if remaining <= 0:
                    raise TimeoutError(
                        "等待屏幕采集超时"
                    )

                self.condition.wait(remaining)

            frame = self.frame
            self.frame = None

        # 主线程取得当前帧后立即请求下一帧；接下来的YOLO推理期间，
        # 截图线程会并行完成下一次grab。
        self.request.set()
        return frame

    def close(self):
        self.stopped.set()
        self.request.set()
        self.thread.join(timeout=1.0)


# ============================================================
# 检测结果
# ============================================================

def detections_from_array(box_data):
    """把Nx6的[x1, y1, x2, y2, conf, class]数组转为检测字典。"""
    detections = []

    for row in box_data:
        x1, y1, x2, y2, confidence, class_id_value = row
        class_id = int(class_id_value)

        if class_id not in CLASS_NAMES:
            continue

        detections.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(confidence),
            "class_id": class_id,
            "track_id": None,
        })

    return detections


def extract_detections(result):
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return []

    # predict模式下每行是[x1, y1, x2, y2, confidence, class_id]。
    # 一次性传回CPU，避免xyxy/conf/cls分别调用cpu()造成三次CUDA同步。
    box_data = (
        boxes.data[:, :6]
        .detach()
        .cpu()
        .numpy()
    )

    return detections_from_array(box_data)


class DirectTorchDetector:
    """
    面向连续小尺寸屏幕帧的轻量PyTorch推理路径。

    model.predict()每帧都会经过Ultralytics Predictor的输入封装、回调
    和Results对象构造。输入只有192×192时，这些固定Python开销可能
    与模型前向本身处于同一量级。这里直接执行模型前向和NMS，只保留
    当前锁敌逻辑真正需要的Nx6检测数据。
    """

    def __init__(
        self,
        yolo_model,
        device,
        imgsz,
        conf,
        iou,
        max_det,
        use_half,
    ):
        self.device = self._resolve_device(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.use_half = bool(
            use_half
            and self.device.type == "cuda"
        )
        self.dtype = (
            torch.float16
            if self.use_half
            else torch.float32
        )

        self.model = yolo_model.model
        self.model.to(self.device)
        self.model.eval()

        if self.use_half:
            self.model.half()
        else:
            self.model.float()

        # 固定输入尺寸下复用CPU预处理数组和GPU输入张量，避免每帧
        # 重新分配多块内存。模型始终串行调用，因此缓冲区可安全复用。
        self.resized_bgr = np.empty(
            (self.imgsz, self.imgsz, 3),
            dtype=np.uint8,
        )
        self.input_chw = np.empty(
            (3, self.imgsz, self.imgsz),
            dtype=np.uint8,
        )
        self.input_cpu_tensor = (
            torch.from_numpy(self.input_chw)
            .unsqueeze(0)
        )
        self.input_tensor = torch.empty(
            (
                1,
                3,
                self.imgsz,
                self.imgsz,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        self.cached_classes_key = None
        self.cached_classes = None

        self._warmup()

    @staticmethod
    def _resolve_device(device):
        device_text = str(device).strip().lower()

        if device_text.isdigit():
            return torch.device(
                f"cuda:{device_text}"
            )

        if device_text == "cuda":
            return torch.device("cuda:0")

        return torch.device(device_text)

    def _warmup(self):
        self.input_tensor.zero_()

        # 首次CUDA调用包含内核加载和算法选择；启动时预热，避免它
        # 被计入运行阶段的帧率，也让cuDNN benchmark完成固定尺寸选择。
        with torch.inference_mode():
            for _ in range(3):
                self.model(self.input_tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def detect(self, frame, classes):
        cv2.resize(
            frame,
            (self.imgsz, self.imgsz),
            dst=self.resized_bgr,
            interpolation=cv2.INTER_LINEAR,
        )

        # BGR HWC -> RGB CHW，直接写入长期复用的连续数组。
        self.input_chw[0] = self.resized_bgr[:, :, 2]
        self.input_chw[1] = self.resized_bgr[:, :, 1]
        self.input_chw[2] = self.resized_bgr[:, :, 0]

        # 复用设备输入张量，省去每帧.cuda()/to()产生的显存分配。
        self.input_tensor.copy_(
            self.input_cpu_tensor,
            non_blocking=False,
        )
        self.input_tensor.mul_(1.0 / 255.0)

        classes_key = tuple(sorted(classes))

        if classes_key != self.cached_classes_key:
            self.cached_classes_key = classes_key
            self.cached_classes = list(classes_key)

        with torch.inference_mode():
            prediction = self.model(self.input_tensor)

            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]

            output = non_max_suppression(
                prediction,
                conf_thres=self.conf,
                iou_thres=self.iou,
                classes=self.cached_classes,
                max_det=self.max_det,
            )[0]

        if output is None or len(output) == 0:
            return []

        # NMS在torch.inference_mode()中返回的是inference tensor。
        # scale_boxes()会对传入的坐标执行原地缩放；新版PyTorch禁止
        # 在InferenceMode外原地修改inference tensor，因此先在普通
        # 模式下clone，得到允许原地更新的常规张量。
        output = output.clone()

        # 当前ROI和推理输入都是正方形，scale_boxes仍用于兼容不同
        # roi-size，并保持坐标映射与Ultralytics实现一致。
        yolo_ops.scale_boxes(
            self.input_tensor.shape[2:],
            output[:, :4],
            frame.shape,
        )

        box_data = (
            output[:, :6]
            .float()
            .cpu()
            .numpy()
        )

        return detections_from_array(box_data)


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
    enable_high_resolution_timer()
    args = parse_args()

    if args.capture_fps <= 0:
        raise ValueError(
            "capture-fps必须大于0"
        )

    if args.debug_fps <= 0:
        raise ValueError(
            "debug-fps必须大于0"
        )

    if args.max_det <= 0:
        raise ValueError(
            "max-det必须大于0"
        )

    frame_interval = 1.0 / args.capture_fps
    debug_interval = 1.0 / args.debug_fps

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到模型：{model_path}"
        )

    model = YOLO(str(model_path))

    # 固定尺寸的连续推理可受益于cuDNN自动选择最快卷积算法。
    # 半精度只在CUDA设备上启用，CPU模式继续使用FP32。
    requested_device = (
        str(args.device).lower()
        if args.device is not None
        else None
    )
    use_cuda = (
        torch.cuda.is_available()
        and requested_device != "cpu"
    )
    use_half = use_cuda

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        model.fuse()
    except (AttributeError, RuntimeError):
        # 某些导出模型或已融合模型不需要再次融合。
        pass

    direct_detector = None

    if (
        args.inference_backend == "direct"
        and model_path.suffix.lower() == ".pt"
    ):
        try:
            direct_detector = DirectTorchDetector(
                yolo_model=model,
                device=args.device,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                use_half=use_half,
            )
        except Exception as exc:
            # 对不符合常规YOLO Detect输出的自定义模型保持兼容性。
            # 回退只发生一次，主循环中不会反复触发异常。
            print(
                "轻量直连推理初始化失败，"
                f"已回退Ultralytics Predictor：{exc}"
            )
            direct_detector = None

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
    parameter_overlay = ParameterOverlay()
    parameter_overlay_enabled = True

    window_name = "YOLO Aim Debug"

    previous_time = time.perf_counter()
    last_debug_time = 0.0
    last_window_poll_time = 0.0
    fps = 0.0
    window_message_interval = 1.0 / WINDOW_MESSAGE_FPS

    print(f"模型：{model_path}")
    print(
        "推理后端："
        + (
            "Direct Torch"
            if direct_detector is not None
            else "Ultralytics Predictor"
        )
    )
    print(f"采集帧率上限：{args.capture_fps:g} FPS")
    print("F1：锁定T")
    print("F2：锁定CT")
    print("F3/F4：降低/提高X轴强度")
    print("F5/F6：降低/提高Y轴强度")
    print("F7/F8：降低/提高平滑强度")
    print("[ / ]：降低/提高锁点区域")
    print("F9/F10：缩小/扩大检测区域")
    print("F11：显示/关闭调试窗口")
    print("HOME：显示/关闭参数")
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

        # 固定参数字典只创建一次；每帧只替换图像和当前阵营类别。
        # 这样可减少高帧率小尺寸推理时的Python端准备开销。
        predict_arguments = {
            "source": None,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "classes": (1, 3),
            "max_det": args.max_det,
            "half": use_half,
            "verbose": False,
        }

        if args.device is not None:
            predict_arguments["device"] = args.device

        screen_capture = PipelinedScreenCapture(
            monitor,
            roi_size,
        )
        next_frame_deadline = time.perf_counter()

        while running:
            # ------------------------------------------------
            # 全局快捷键
            # ------------------------------------------------

            if hotkeys.pressed_once(VK_END):
                break

            if hotkeys.pressed_once(VK_HOME):
                parameter_overlay_enabled = (
                    not parameter_overlay_enabled
                )

                if not parameter_overlay_enabled:
                    parameter_overlay.close()

                print(
                    "参数显示："
                    + (
                        "开启"
                        if parameter_overlay_enabled
                        else "关闭"
                    )
                )

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
                screen_capture.set_roi_size(roi_size)

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
                screen_capture.set_roi_size(roi_size)

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

            frame = screen_capture.get_frame()

            # ------------------------------------------------
            # YOLO检测
            # ------------------------------------------------

            detections = None

            if direct_detector is not None:
                try:
                    detections = direct_detector.detect(
                        frame,
                        target_classes,
                    )
                except Exception as exc:
                    print(
                        "轻量直连推理运行失败，"
                        f"已回退Ultralytics Predictor：{exc}"
                    )
                    direct_detector = None

            if detections is None:
                predict_arguments["source"] = frame
                predict_arguments["classes"] = tuple(
                    sorted(target_classes)
                )

                # 当前锁敌逻辑每帧按准星距离重新选目标，不读取Track ID。
                # 因此不再运行ByteTrack，避免无效的追踪开销。
                results = model.predict(
                    **predict_arguments
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
            # 独立参数悬浮面板
            # ------------------------------------------------

            if parameter_overlay_enabled:
                parameter_overlay.show(
                    target_mode=target_mode,
                    x_strength=x_strength,
                    y_strength=y_strength,
                    smoothing=smoothing,
                    aim_region=aim_region,
                    roi_size=roi_size,
                    aiming=aiming,
                )

            # ------------------------------------------------
            # 调试窗口
            # ------------------------------------------------

            if debug_enabled:
                debug_now = time.perf_counter()

                # 调试画面独立限帧，避免frame.copy、框线和文字绘制
                # 阻塞识别与控制循环。窗口消息仍在每帧处理。
                if (
                    debug_now - last_debug_time
                    >= debug_interval
                ):
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
                    last_debug_time = debug_now

                # 窗口消息最高按60 Hz处理；识别循环达到90 FPS时无需
                # 额外执行90次GUI调用。pollKey本身不主动等待。
                key = -1

                if (
                    debug_now - last_window_poll_time
                    >= window_message_interval
                ):
                    key = poll_window_key()
                    last_window_poll_time = debug_now

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    running = False

            else:
                window_now = time.perf_counter()

                if (
                    window_now - last_window_poll_time
                    >= window_message_interval
                ):
                    poll_window_key()
                    last_window_poll_time = window_now

            # OpenCV刷新其他窗口后，再把参数层放回最上层。
            # 这一步放在循环末尾，避免调试窗口的imshow/waitKey
            # 改变参数层的Z轴顺序。
            if parameter_overlay_enabled:
                parameter_overlay.keep_topmost()

            # 使用固定时间轴限帧。旧实现从每帧开始重新计算截止点，
            # Windows sleep每次多睡的一点时间会不断累积，使目标FPS
            # 长期略低于设定值。固定时间轴会在下一帧自动
            # 补偿少量超时；处理本身超过目标间隔时则立即重新同步。
            next_frame_deadline += frame_interval
            frame_finished_at = time.perf_counter()

            if next_frame_deadline > frame_finished_at:
                wait_until(next_frame_deadline)
            else:
                next_frame_deadline = frame_finished_at

        screen_capture.close()

    parameter_overlay.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()