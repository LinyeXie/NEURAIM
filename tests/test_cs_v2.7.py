import argparse
import atexit
import ctypes
import math
import threading
import time
from collections import deque
from ctypes import wintypes
from pathlib import Path

import cv2
import mss
import numpy as np
import torch
from ultralytics import YOLO



try:
    import interception
except ImportError:
    interception = None


class InterceptionMouseBackend:
    def __init__(self):
        self.available = interception is not None

    def move_relative(self, dx, dy):
        if not self.available:
            return False

        if hasattr(interception, "move_relative"):
            interception.move_relative(int(dx), int(dy))
            return True

        if hasattr(interception, "move"):
            interception.move(int(dx), int(dy))
            return True

        raise RuntimeError(
            "当前 interception 包没有找到相对移动接口，请检查安装版本"
        )

    def left_click(self):
        if not self.available:
            return False

        if hasattr(interception, "click"):
            interception.click()
            return True

        if hasattr(interception, "mouse_down") and hasattr(interception, "mouse_up"):
            interception.mouse_down("left")
            interception.mouse_up("left")
            return True

        raise RuntimeError(
            "当前 interception 包没有找到点击接口，请检查安装版本"
        )


mouse_backend = InterceptionMouseBackend()

try:
   
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    
    from ultralytics.utils.ops import non_max_suppression


# ============================================================
# 路径与类别
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = Path(
    PROJECT_ROOT / "runs" / "yolo12_l" / "yolo12l_.engine"
)

CLASS_NAMES = {
    0: "c",
    1: "ch",
    2: "t",
    3: "th",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 255, 0),
    2: (0, 0, 255),
    3: (0, 165, 255),
}

TEAM_PART_CLASSES = {
    "c": {
        "body": 0,
        "head": 1,
    },
    "t": {
        "body": 2,
        "head": 3,
    },
}

DEFAULT_CAPTURE_FPS = 120.0
DEFAULT_DEBUG_FPS = 30.0
DEFAULT_MAX_DET = 32
WINDOW_MESSAGE_FPS = 30.0
HOTKEY_POLL_FPS = 30.0
DEFAULT_TRIGGER_RANGE = 3
DEFAULT_TRIGGER_INTERVAL_MS = 20
TRIGGER_INTERVAL_STEP_MS = 10

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
VK_F12 = 0x7B

# [ 和 ]
VK_OEM_4 = 0xDB
VK_OEM_6 = 0xDD

# 主键盘 :、"、< 和 >
# Windows对OEM标点键返回的是物理键虚拟码；不额外检查Shift，
# 因而可兼容中英文输入法以及不同键盘布局下的同一按键位置。
VK_OEM_1 = 0xBA
VK_OEM_7 = 0xDE
VK_OEM_COMMA = 0xBC
VK_OEM_PERIOD = 0xBE

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

user32 = ctypes.windll.user32


# ============================================================
# 独立参数悬浮面板
# ============================================================

class ParameterOverlay:


    PANEL_WIDTH = 430
    PANEL_HEIGHT = 316


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
    
    TOPMOST_REFRESH_INTERVAL = 1.00

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
        auto_trigger_enabled,
        trigger_range,
        trigger_interval_ms,
        trigger_in_range,
        debug_enabled,
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
            bool(auto_trigger_enabled),
            int(trigger_range),
            int(trigger_interval_ms),
            bool(trigger_in_range),
            bool(debug_enabled),
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
            "T (2) / TH (3)"
            if target_mode == "t"
            else "C (0) / CH (1)"
        )
        trigger_text = (
            "OFF"
            if not auto_trigger_enabled
            else (
                "FIRE"
                if trigger_in_range
                else "ON"
            )
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
            (
                f"Auto trigger [F11] : {trigger_text}",
                (
                    (0, 255, 0)
                    if trigger_in_range
                    else (
                        (0, 220, 255)
                        if auto_trigger_enabled
                        else (170, 170, 170)
                    )
                ),
            ),
            (
                f'Trigger range [: / "] : {trigger_range} px',
                (255, 255, 255),
            ),
            (
                f"Click interval [< / >] : "
                f"{trigger_interval_ms} ms",
                (255, 255, 255),
            ),
            (
                f"Debug window [F12] : "
                f"{'ON' if debug_enabled else 'OFF'}",
                (255, 255, 255),
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



class AutoTriggerController:


    def __init__(self):
        self.last_click_time = 0.0
        self.in_range = False

    def reset(self):
        self.last_click_time = 0.0
        self.in_range = False

    @staticmethod
    def _send_left_click():
        return mouse_backend.left_click()


    def update(
        self,
        enabled,
        aim_point,
        center_x,
        center_y,
        trigger_range,
        trigger_interval_ms,
        measurement_valid,
    ):
        if (
            not enabled
            or aim_point is None
            or not measurement_valid
        ):
            self.reset()
            return False

        distance = math.hypot(
            aim_point[0] - center_x,
            aim_point[1] - center_y,
        )
        self.in_range = distance <= trigger_range

        if not self.in_range:
            self.last_click_time = 0.0
            return False

        now = time.perf_counter()
        interval_seconds = trigger_interval_ms / 1000.0

        if (
            self.last_click_time == 0.0
            or now - self.last_click_time
            >= interval_seconds
        ):
            if self._send_left_click():
                self.last_click_time = now

        return True




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
    设定帧率的情况。最后约0.3毫秒短暂忙等，避免Sleep(0)在
    120 FPS的8.33毫秒帧预算中被Windows调度器明显延后。
    """
    while True:
        remaining = deadline - time.perf_counter()

        if remaining <= 0:
            return

        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        elif remaining > 0.0003:
            time.sleep(0)
        else:
            while time.perf_counter() < deadline:
                pass
            return


class HotkeyState:
    """将全局按键转换成单次按下事件。"""

    def __init__(self):
        self.previous = {}
        self.last_poll = {}
        self.poll_interval = 1.0 / HOTKEY_POLL_FPS
        self.frame_time = 0.0

    def begin_frame(self):

        self.frame_time = time.perf_counter()

    def pressed_once(self, vk_code):
        now = (
            self.frame_time
            if self.frame_time > 0.0
            else time.perf_counter()
        )
        last_poll = self.last_poll.get(vk_code, 0.0)


        if now - last_poll < self.poll_interval:
            return False

        self.last_poll[vk_code] = now
        current = is_key_down(vk_code)
        previous = self.previous.get(vk_code, False)

        self.previous[vk_code] = current

        return current and not previous



class AimController:

    COMMAND_HISTORY_SECONDS = 0.30

    COMMAND_VISIBILITY_DELAY = 0.010

    MAX_LEAD_SECONDS = 0.085
    MAX_TARGET_SPEED = 2200.0

    def __init__(self):
        self.command_history = deque()
        self.motion_samples_x = deque(maxlen=12)
        self.motion_samples_y = deque(maxlen=12)

        self.velocity_samples_x = deque(maxlen=5)
        self.velocity_samples_y = deque(maxlen=5)

        self.response_x = 1.0
        self.response_y = 1.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.filtered_output_x = 0.0
        self.filtered_output_y = 0.0
        self.residual_x = 0.0
        self.residual_y = 0.0

        self.previous_error_x = 0.0
        self.previous_error_y = 0.0
        self.previous_capture_time = 0.0
        self.last_frame_interval = 1.0 / DEFAULT_CAPTURE_FPS

        self.reverse_votes_x = 0
        self.reverse_votes_y = 0
        self.last_sent_x = 0
        self.last_sent_y = 0
        self.previous_control_error_x = 0.0
        self.previous_control_error_y = 0.0
        self.initialized = False

    def reset(self):
        self.command_history.clear()
        self.motion_samples_x.clear()
        self.motion_samples_y.clear()
        self.velocity_samples_x.clear()
        self.velocity_samples_y.clear()

        self.response_x = 1.0
        self.response_y = 1.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.filtered_output_x = 0.0
        self.filtered_output_y = 0.0
        self.residual_x = 0.0
        self.residual_y = 0.0

        self.previous_error_x = 0.0
        self.previous_error_y = 0.0
        self.previous_capture_time = 0.0
        self.last_frame_interval = 1.0 / DEFAULT_CAPTURE_FPS

        self.reverse_votes_x = 0
        self.reverse_votes_y = 0
        self.last_sent_x = 0
        self.last_sent_y = 0
        self.previous_control_error_x = 0.0
        self.previous_control_error_y = 0.0
        self.initialized = False

    def _trim_command_history(self, now):
        oldest = now - self.COMMAND_HISTORY_SECONDS

        while (
            self.command_history
            and self.command_history[0][0] < oldest
        ):
            self.command_history.popleft()

    def _sum_commands(self, start_time, end_time):
        total_x = 0.0
        total_y = 0.0

        for sent_at, move_x, move_y in self.command_history:
            if sent_at <= start_time:
                continue

            if sent_at > end_time:
                break

            total_x += move_x
            total_y += move_y

        return total_x, total_y

    def project_screen_point(
        self,
        point_x,
        point_y,
        start_time,
        end_time,
    ):
 
        if start_time <= 0.0 or end_time <= start_time:
            return point_x, point_y

        visible_until = (
            end_time - self.COMMAND_VISIBILITY_DELAY
        )
        move_x, move_y = self._sum_commands(
            start_time - self.COMMAND_VISIBILITY_DELAY,
            visible_until,
        )
        dt = clamp(end_time - start_time, 0.0, 0.050)

        return (
            point_x
            - move_x * self.response_x
            + self.velocity_x * dt,
            point_y
            - move_y * self.response_y
            + self.velocity_y * dt,
        )

    @staticmethod
    def _fit_response(samples, current_response):

        if len(samples) < 7:
            return current_response

        sum_dt2 = 0.0
        sum_dt_mouse = 0.0
        sum_mouse2 = 0.0
        sum_dt_delta = 0.0
        sum_mouse_delta = 0.0

        for dt, mouse_delta, error_delta in samples:
            sum_dt2 += dt * dt
            sum_dt_mouse += dt * mouse_delta
            sum_mouse2 += mouse_delta * mouse_delta
            sum_dt_delta += dt * error_delta
            sum_mouse_delta += mouse_delta * error_delta

        # 设计矩阵列为[dt, -mouse_delta]。
        a00 = sum_dt2
        a01 = -sum_dt_mouse
        a11 = sum_mouse2
        b0 = sum_dt_delta
        b1 = -sum_mouse_delta
        determinant = a00 * a11 - a01 * a01

        if determinant <= 1e-7 or sum_mouse2 < 20.0:
            return current_response

        estimated_response = (
            a00 * b1 - a01 * b0
        ) / determinant


        if not 0.08 <= estimated_response <= 6.00:
            return current_response


        return (
            current_response * 0.90
            + estimated_response * 0.10
        )

    @staticmethod
    def _robust_velocity(
        velocity_samples,
        previous_velocity,
        dt,
        smoothing,
    ):
        if not velocity_samples:
            return previous_velocity

        values = np.asarray(
            tuple(velocity_samples)[-5:],
            dtype=np.float32,
        )
        recent_values = values[-3:]
        median_velocity = float(np.median(recent_values))


        if abs(median_velocity) < 20.0:
            median_velocity = 0.0

        median_velocity = clamp(
            median_velocity,
            -AimController.MAX_TARGET_SPEED,
            AimController.MAX_TARGET_SPEED,
        )


        smooth_curve = (
            smoothing
            * smoothing
            * (3.0 - 2.0 * smoothing)
        )
        time_constant = 0.008 + smooth_curve * 0.022
        alpha = 1.0 - math.exp(
            -dt / max(0.001, time_constant)
        )


        if (
            previous_velocity != 0.0
            and (
                median_velocity * previous_velocity <= 0.0
                or abs(median_velocity)
                < abs(previous_velocity)
            )
        ):
            alpha = max(alpha, 0.85)

        return previous_velocity + (
            median_velocity - previous_velocity
        ) * alpha

    def _update_motion_estimate(
        self,
        error_x,
        error_y,
        capture_time,
        target_scale,
        smoothing,
    ):
        if not self.initialized:
            self.previous_error_x = error_x
            self.previous_error_y = error_y
            self.previous_capture_time = capture_time
            self.initialized = True
            return

        dt = capture_time - self.previous_capture_time

        if not 0.001 <= dt <= 0.080:
            self.previous_error_x = error_x
            self.previous_error_y = error_y
            self.previous_capture_time = capture_time
            self.velocity_samples_x.clear()
            self.velocity_samples_y.clear()
            self.velocity_x *= 0.5
            self.velocity_y *= 0.5
            return

        self.last_frame_interval = clamp(
            dt,
            1.0 / 240.0,
            1.0 / 30.0,
        )

        visible_start = (
            self.previous_capture_time
            - self.COMMAND_VISIBILITY_DELAY
        )
        visible_end = (
            capture_time
            - self.COMMAND_VISIBILITY_DELAY
        )
        applied_x, applied_y = self._sum_commands(
            visible_start,
            visible_end,
        )

        delta_x = error_x - self.previous_error_x
        delta_y = error_y - self.previous_error_y

        self.motion_samples_x.append(
            (dt, applied_x, delta_x)
        )
        self.motion_samples_y.append(
            (dt, applied_y, delta_y)
        )

        self.response_x = self._fit_response(
            self.motion_samples_x,
            self.response_x,
        )
        self.response_y = self._fit_response(
            self.motion_samples_y,
            self.response_y,
        )

        external_delta_x = (
            delta_x + applied_x * self.response_x
        )
        external_delta_y = (
            delta_y + applied_y * self.response_y
        )


        max_external_delta = max(
            3.0,
            target_scale * 0.30,
        )
        external_delta_x = clamp(
            external_delta_x,
            -max_external_delta,
            max_external_delta,
        )
        external_delta_y = clamp(
            external_delta_y,
            -max_external_delta,
            max_external_delta,
        )

        self.velocity_samples_x.append(
            external_delta_x / dt
        )
        self.velocity_samples_y.append(
            external_delta_y / dt
        )

        self.velocity_x = self._robust_velocity(
            self.velocity_samples_x,
            self.velocity_x,
            dt,
            smoothing,
        )
        self.velocity_y = self._robust_velocity(
            self.velocity_samples_y,
            self.velocity_y,
            dt,
            smoothing,
        )

        self.previous_error_x = error_x
        self.previous_error_y = error_y
        self.previous_capture_time = capture_time

    @staticmethod
    def _soft_deadzone(value, radius):
        magnitude = abs(value)

        if magnitude <= radius:
            return 0.0

        return math.copysign(
            magnitude - radius,
            value,
        )

    @staticmethod
    def _limit_lead(velocity, latency, target_scale):
        lead = velocity * latency
        lead_limit = clamp(
            target_scale * 0.55,
            8.0,
            42.0,
        )
        return clamp(lead, -lead_limit, lead_limit)

    @staticmethod
    def _motion_confidence(samples, velocity):

        if len(samples) < 3 or abs(velocity) < 18.0:
            return 0.0

        values = np.asarray(
            tuple(samples)[-5:],
            dtype=np.float32,
        )
        direction = 1.0 if velocity > 0.0 else -1.0

        # 最新实测已经停止或反向时，不等待整个中位数窗口刷新。
        if (
            abs(float(values[-1])) < 16.0
            or float(values[-1]) * direction <= 0.0
        ):
            return 0.0

        same_direction = float(
            np.mean(values * direction > 8.0)
        )

        if same_direction < 0.60:
            return 0.0

        median = float(np.median(values))
        median_magnitude = abs(median)

        if median_magnitude < 16.0:
            return 0.0

        mad = float(
            np.median(np.abs(values - median))
        )
        direction_score = clamp(
            (same_direction - 0.60) / 0.35,
            0.0,
            1.0,
        )
        noise_score = 1.0 - clamp(
            mad / max(45.0, median_magnitude * 0.85),
            0.0,
            1.0,
        )
        return direction_score * noise_score

    @staticmethod
    def _smooth_output(requested, previous, smoothing):
        if requested == 0.0:
            return 0.0

        # 反向时不携带旧方向惯性；直接从新的小指令开始。
        if requested * previous <= 0.0:
            return requested


        smooth_curve = (
            smoothing
            * smoothing
            * (3.0 - 2.0 * smoothing)
        )
        alpha = (
            1.0 - smooth_curve * 0.86
        )
        output = previous + (
            requested - previous
        ) * alpha

  
        if abs(output) > abs(requested):
            output = requested

        return output

    @staticmethod
    def _guard_reversal(
        requested,
        predicted_error,
        last_sent,
        reverse_votes,
        deadzone,
    ):
        near_lock = (
            abs(predicted_error)
            < max(2.5, deadzone * 1.8)
        )
        reversing = (
            requested != 0.0
            and last_sent != 0
            and requested * last_sent < 0.0
        )

        if reversing and near_lock:
            reverse_votes += 1

 
            if reverse_votes == 1:
                return 0.0, reverse_votes

            return requested * 0.40, reverse_votes

        return requested, 0

    def _send_mouse(
        self,
        move_x,
        move_y,
        logical_x=None,
        logical_y=None,
    ):
        if logical_x is None:
            logical_x = move_x

        if logical_y is None:
            logical_y = move_y

        if move_x == 0 and move_y == 0:
            return 0, 0

        if not mouse_backend.move_relative(move_x, move_y):
            self.last_sent_x = 0
            self.last_sent_y = 0
            return 0, 0

        sent_at = time.perf_counter()
        self.command_history.append(
            (sent_at, logical_x, logical_y)
        )
        self._trim_command_history(sent_at)
        if logical_x != 0:
            self.last_sent_x = logical_x

        if logical_y != 0:
            self.last_sent_y = logical_y

        return move_x, move_y

    def update(
        self,
        aim_x,
        aim_y,
        center_x,
        center_y,
        capture_time,
        target_scale,
        x_strength,
        y_strength,
        smoothing,
        max_step,
        deadzone,
        invert_y,
        measurement_valid=True,
    ):
 
        now = time.perf_counter()
        self._trim_command_history(now)

        measured_error_x = aim_x - center_x
        measured_error_y = aim_y - center_y

        if measurement_valid:
            self._update_motion_estimate(
                measured_error_x,
                measured_error_y,
                capture_time,
                target_scale,
                smoothing,
            )
        elif self.initialized:

            self.velocity_x *= 0.72
            self.velocity_y *= 0.72

        pending_start = (
            capture_time - self.COMMAND_VISIBILITY_DELAY
        )
        pending_x, pending_y = self._sum_commands(
            pending_start,
            now,
        )

        predicted_error_x = (
            measured_error_x
            - pending_x * self.response_x
        )
        predicted_error_y = (
            measured_error_y
            - pending_y * self.response_y
        )


        if measured_error_x * predicted_error_x < 0.0:
            predicted_error_x = 0.0

        if measured_error_y * predicted_error_y < 0.0:
            predicted_error_y = 0.0

        error_radius = math.hypot(
            predicted_error_x,
            predicted_error_y,
        )
        stop_radius = max(0.75, deadzone)


        confidence_x = (
            self._motion_confidence(
                self.velocity_samples_x,
                self.velocity_x,
            )
            if measurement_valid
            else 0.0
        )
        confidence_y = (
            self._motion_confidence(
                self.velocity_samples_y,
                self.velocity_y,
            )
            if measurement_valid
            else 0.0
        )
        motion_active = (
            abs(self.velocity_x * confidence_x) >= 18.0
            or abs(self.velocity_y * confidence_y) >= 18.0
        )


        if error_radius <= stop_radius and not motion_active:
            self.filtered_output_x = 0.0
            self.filtered_output_y = 0.0
            self.residual_x = 0.0
            self.residual_y = 0.0
            self.reverse_votes_x = 0
            self.reverse_votes_y = 0
            self.last_sent_x = 0
            self.last_sent_y = 0
            self.previous_control_error_x = predicted_error_x
            self.previous_control_error_y = predicted_error_y

            return {
                "move_x": 0,
                "move_y": 0,
                "predicted_error_x": predicted_error_x,
                "predicted_error_y": predicted_error_y,
                "velocity_x": self.velocity_x,
                "velocity_y": self.velocity_y,
            }

        if error_radius > stop_radius:
            radial_scale = (
                error_radius - stop_radius
            ) / error_radius
            radial_error_x = predicted_error_x * radial_scale
            radial_error_y = predicted_error_y * radial_scale
        else:
            radial_error_x = 0.0
            radial_error_y = 0.0


        reference_interval = 1.0 / DEFAULT_CAPTURE_FPS
        frame_scale = clamp(
            self.last_frame_interval / reference_interval,
            0.50,
            4.00,
        )
        gain_x = 1.0 - math.exp(
            -max(0.0, x_strength) * frame_scale
        )
        gain_y = 1.0 - math.exp(
            -max(0.0, y_strength) * frame_scale
        )


        brake_radius = clamp(
            target_scale * 0.16,
            7.0,
            22.0,
        )
        brake_progress = clamp(
            (error_radius - stop_radius)
            / max(1.0, brake_radius),
            0.0,
            1.0,
        )
        brake_factor = (
            0.55
            + 0.45
            * brake_progress
            * brake_progress
            * (3.0 - 2.0 * brake_progress)
        )
        smooth_curve = (
            smoothing
            * smoothing
            * (3.0 - 2.0 * smoothing)
        )
        damping = (
            1.0 - smooth_curve * 0.26
        )

        position_screen_x = (
            radial_error_x
            * gain_x
            * brake_factor
            * damping
        )
        position_screen_y = (
            radial_error_y
            * gain_y
            * brake_factor
            * damping
        )

        if not measurement_valid:

            position_screen_x = 0.0
            position_screen_y = 0.0


        measurement_age = clamp(
            now - capture_time,
            0.0,
            0.050,
        )
        lead_time = clamp(
            measurement_age
            + self.COMMAND_VISIBILITY_DELAY
            + self.last_frame_interval * 1.15,
            0.0,
            self.MAX_LEAD_SECONDS,
        )

        lead_gain_x = clamp(
            0.22 + max(0.0, x_strength) * 0.18,
            0.0,
            0.76,
        )
        lead_gain_y = clamp(
            0.22 + max(0.0, y_strength) * 0.18,
            0.0,
            0.76,
        )
        lead_screen_x = self._limit_lead(
            self.velocity_x * confidence_x,
            lead_time,
            target_scale,
        ) * lead_gain_x
        lead_screen_y = self._limit_lead(
            self.velocity_y * confidence_y,
            lead_time,
            target_scale,
        ) * lead_gain_y


        if (
            confidence_x > 0.0
            and predicted_error_x * self.velocity_x > 0.0
        ):
            catchup_time_x = min(
                self.last_frame_interval * 0.75,
                abs(predicted_error_x)
                / max(1.0, abs(self.velocity_x))
                * 0.35,
                0.018,
            )
            lead_screen_x += (
                self.velocity_x
                * confidence_x
                * catchup_time_x
                * clamp(x_strength / 3.0, 0.0, 1.0)
            )

        if (
            confidence_y > 0.0
            and predicted_error_y * self.velocity_y > 0.0
        ):
            catchup_time_y = min(
                self.last_frame_interval * 0.75,
                abs(predicted_error_y)
                / max(1.0, abs(self.velocity_y))
                * 0.35,
                0.018,
            )
            lead_screen_y += (
                self.velocity_y
                * confidence_y
                * catchup_time_y
                * clamp(y_strength / 3.0, 0.0, 1.0)
            )

        total_lead_limit = clamp(
            target_scale * 0.70,
            10.0,
            50.0,
        )
        lead_screen_x = clamp(
            lead_screen_x,
            -total_lead_limit,
            total_lead_limit,
        )
        lead_screen_y = clamp(
            lead_screen_y,
            -total_lead_limit,
            total_lead_limit,
        )

        desired_screen_x = (
            position_screen_x + lead_screen_x
        )
        desired_screen_y = (
            position_screen_y + lead_screen_y
        )


        guidance_x = predicted_error_x + lead_screen_x
        guidance_y = predicted_error_y + lead_screen_y
        guidance_radius = math.hypot(
            guidance_x,
            guidance_y,
        )

        if guidance_radius <= 1e-6:
            desired_screen_x = 0.0
            desired_screen_y = 0.0
            unit_x = 0.0
            unit_y = 0.0
        else:
            unit_x = guidance_x / guidance_radius
            unit_y = guidance_y / guidance_radius

        radial_command = (
            desired_screen_x * unit_x
            + desired_screen_y * unit_y
        )
        tangent_x = (
            desired_screen_x
            - radial_command * unit_x
        )
        tangent_y = (
            desired_screen_y
            - radial_command * unit_y
        )
        radial_command = max(0.0, radial_command)
        tangent_length = math.hypot(
            tangent_x,
            tangent_y,
        )
        position_length = math.hypot(
            position_screen_x,
            position_screen_y,
        )
        tangent_limit = max(
            1.25,
            position_length * 0.65,
            math.hypot(
                lead_screen_x,
                lead_screen_y,
            ) * 0.30,
        )

        if tangent_length > tangent_limit:
            tangent_scale = tangent_limit / tangent_length
            tangent_x *= tangent_scale
            tangent_y *= tangent_scale

        desired_screen_x = (
            radial_command * unit_x + tangent_x
        )
        desired_screen_y = (
            radial_command * unit_y + tangent_y
        )

        effective_response_x = clamp(
            self.response_x,
            0.08,
            6.00,
        )
        effective_response_y = clamp(
            self.response_y,
            0.08,
            6.00,
        )
        requested_x = (
            desired_screen_x / effective_response_x
        )
        requested_y = (
            desired_screen_y / effective_response_y
        )


        if (
            self.previous_control_error_x != 0.0
            and guidance_x
            * self.previous_control_error_x <= 0.0
        ):
            self.residual_x = 0.0
            requested_x *= 0.35

        if (
            self.previous_control_error_y != 0.0
            and guidance_y
            * self.previous_control_error_y <= 0.0
        ):
            self.residual_y = 0.0
            requested_y *= 0.35

        self.previous_control_error_x = guidance_x
        self.previous_control_error_y = guidance_y

        requested_x = self._smooth_output(
            requested_x,
            self.filtered_output_x,
            smoothing,
        )
        requested_y = self._smooth_output(
            requested_y,
            self.filtered_output_y,
            smoothing,
        )

        calibration_progress = clamp(
            min(
                len(self.motion_samples_x),
                len(self.motion_samples_y),
            ) / 8.0,
            0.0,
            1.0,
        )
        safe_frame_scale = (
            1.0
            + (frame_scale - 1.0)
            * calibration_progress
        )
        effective_max_step = (
            max_step * safe_frame_scale
            if max_step > 0.0
            else 0.0
        )

        requested_length = math.hypot(
            requested_x,
            requested_y,
        )
        if (
            requested_length > effective_max_step > 0.0
        ):
            vector_scale = (
                effective_max_step / requested_length
            )
            requested_x *= vector_scale
            requested_y *= vector_scale

        self.filtered_output_x = requested_x
        self.filtered_output_y = requested_y

        if requested_x == 0.0:
            self.residual_x = 0.0
        elif self.residual_x * requested_x < 0.0:
            self.residual_x = 0.0

        if requested_y == 0.0:
            self.residual_y = 0.0
        elif self.residual_y * requested_y < 0.0:
            self.residual_y = 0.0

        self.residual_x += requested_x
        self.residual_y += requested_y

        move_x = int(self.residual_x)
        logical_move_y = int(self.residual_y)
        self.residual_x -= move_x
        self.residual_y -= logical_move_y

        minimum_pulse = 0.30

        if (
            move_x == 0
            and abs(requested_x) >= minimum_pulse
            and abs(guidance_x)
            > max(
                stop_radius * 1.50,
                effective_response_x * 1.10,
            )
        ):
            move_x = 1 if requested_x > 0.0 else -1
            self.residual_x = 0.0

        if (
            logical_move_y == 0
            and abs(requested_y) >= minimum_pulse
            and abs(guidance_y)
            > max(
                stop_radius * 1.50,
                effective_response_y * 1.10,
            )
        ):
            logical_move_y = (
                1 if requested_y > 0.0 else -1
            )
            self.residual_y = 0.0

        screen_dot_x = (
            move_x
            * effective_response_x
            * guidance_x
        )
        screen_dot_y = (
            logical_move_y
            * effective_response_y
            * guidance_y
        )

        if screen_dot_x + screen_dot_y < 0.0:
            if screen_dot_x > 0.0:
                logical_move_y = 0
                self.residual_y = 0.0
            elif screen_dot_y > 0.0:
                move_x = 0
                self.residual_x = 0.0
            else:
                move_x = 0
                logical_move_y = 0
                self.residual_x = 0.0
                self.residual_y = 0.0

        raw_move_y = (
            -logical_move_y
            if invert_y
            else logical_move_y
        )
        sent_x, sent_y = self._send_mouse(
            move_x,
            raw_move_y,
            logical_x=move_x,
            logical_y=logical_move_y,
        )

        return {
            "move_x": sent_x,
            "move_y": sent_y,
            "predicted_error_x": predicted_error_x,
            "predicted_error_y": predicted_error_y,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
        }


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
        default=512,
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
        help="采集与识别循环帧率上限，默认120 FPS",
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
        default=1.8,
        help="X轴锁定强度，建议0.8～3.0；同时控制追赶与动态维持",
    )

    parser.add_argument(
        "--y-strength",
        type=float,
        default=1.8,
        help="Y轴锁定强度，建议0.8～3.0；同时控制追赶与动态维持",
    )

    parser.add_argument(
        "--smooth",
        type=float,
        default=0.20,
        help="平滑强度，范围0.00～1.00",
    )

    parser.add_argument(
        "--aim-region",
        type=float,
        default=1.0,
        help="锁点区域：0为元素身中心，1为元素头中心",
    )

    parser.add_argument(
        "--max-step",
        type=float,
        default=100.0,
        help=(
            "120 FPS参考下的单帧二维最大移动量；"
            "实际运行时会按帧间隔自动补偿"
        ),
    )

    parser.add_argument(
        "--deadzone",
        type=float,
        default=1.0,
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


    def __init__(self, monitor, roi_size):
        self.monitor = dict(monitor)
        self.roi_size = int(roi_size)

        self.condition = threading.Condition()
        self.request = threading.Event()
        self.stopped = threading.Event()

        self.frame = None
        self.frame_time = 0.0
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

            buffer_generation = -1
            frame_buffers = None
            buffer_index = 0

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
                    captured_at = time.perf_counter()

                    if (
                        frame_buffers is None
                        or buffer_generation != generation
                    ):
                        frame_buffers = (
                            np.empty(
                                (roi_size, roi_size, 3),
                                dtype=np.uint8,
                            ),
                            np.empty(
                                (roi_size, roi_size, 3),
                                dtype=np.uint8,
                            ),
                        )
                        buffer_generation = generation
                        buffer_index = 0

                    # MSS返回BGRA。直接把BGR三通道复制进可复用缓冲，
                    # 避免np.ascontiguousarray每帧创建新数组。
                    frame = frame_buffers[buffer_index]
                    np.copyto(
                        frame,
                        np.asarray(screenshot)[:, :, :3],
                    )
                    buffer_index ^= 1

                    with self.condition:
                        # 检测区域在采集期间发生变化时丢弃旧尺寸帧。
                        if generation == self.generation:
                            self.frame = frame
                            self.frame_time = captured_at
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
            self.frame_time = 0.0
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
            frame_time = self.frame_time
            self.frame = None

        # 主线程取得当前帧后立即请求下一帧；接下来的YOLO推理期间，
        # 截图线程会并行完成下一次grab。
        self.request.set()
        return frame, frame_time

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
        # CUDA模式优先使用锁页内存，减少每帧CPU->GPU拷贝延迟。
        # 某些精简PyTorch构建不支持pin_memory，因此保留普通内存回退。
        self.use_pinned_memory = False

        if self.device.type == "cuda":
            try:
                self.input_cpu_tensor = torch.empty(
                    (
                        1,
                        3,
                        self.imgsz,
                        self.imgsz,
                    ),
                    dtype=torch.uint8,
                    pin_memory=True,
                )
                self.use_pinned_memory = True
            except (RuntimeError, NotImplementedError):
                self.input_cpu_tensor = torch.empty(
                    (
                        1,
                        3,
                        self.imgsz,
                        self.imgsz,
                    ),
                    dtype=torch.uint8,
                )
        else:
            self.input_cpu_tensor = torch.empty(
                (
                    1,
                    3,
                    self.imgsz,
                    self.imgsz,
                ),
                dtype=torch.uint8,
            )

        self.input_chw = self.input_cpu_tensor[0].numpy()
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
        self.cached_frame_shape = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.cuda_graph = None
        self.cuda_graph_prediction = None

        self._warmup()
        # CUDA Graph已禁用，始终使用普通CUDA直连推理。

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

    def _try_enable_cuda_graph(self):
        """
        为固定输入形状录制模型前向。

        NMS仍在Graph外执行，因此类别、置信度和IoU参数可以随程序
        正常工作。只录制最稳定、开销最大的模型前向；任何兼容性
        问题都在这里回退，不影响Direct Torch继续运行。
        """
        if self.device.type != "cuda":
            return

        graph = None

        try:
            # CUDA Graph要求在非默认流上先完成充分预热，以便卷积
            # 算法选择、工作区及缓存分配都在正式录制前结束。
            warmup_stream = torch.cuda.Stream(
                device=self.device,
            )
            warmup_stream.wait_stream(
                torch.cuda.current_stream(self.device)
            )

            with torch.cuda.stream(warmup_stream):
                with torch.inference_mode():
                    for _ in range(3):
                        self.model(self.input_tensor)

            torch.cuda.current_stream(
                self.device
            ).wait_stream(warmup_stream)
            torch.cuda.synchronize(self.device)

            graph = torch.cuda.CUDAGraph()

            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    self.cuda_graph_prediction = self.model(
                        self.input_tensor
                    )

            # 立即重放一次，确保录制结果在进入实时循环前可用。
            graph.replay()
            torch.cuda.synchronize(self.device)
            self.cuda_graph = graph
            print("CUDA Graph前向：已启用")
        except Exception as exc:
            self.cuda_graph = None
            self.cuda_graph_prediction = None

            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass

            print(
                "CUDA Graph前向不可用，"
                f"继续使用普通直连推理：{exc}"
            )

    @torch.inference_mode()
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
            non_blocking=self.use_pinned_memory,
        )
        self.input_tensor.mul_(1.0 / 255.0)

        if classes != self.cached_classes_key:
            self.cached_classes_key = classes
            self.cached_classes = list(classes)

        if self.cuda_graph is not None:
            self.cuda_graph.replay()
            prediction = self.cuda_graph_prediction
        else:
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

        # 只进行一次GPU->CPU同步，然后在很小的Nx6数组上缩放坐标。
        # 这省去GPU端clone、scale_boxes的小算子，也彻底规避新版
        # PyTorch对inference tensor原地更新的限制。
        box_data = (
            output[:, :6]
            .cpu()
            .float()
            .numpy()
        )

        frame_shape = frame.shape[:2]

        if frame_shape != self.cached_frame_shape:
            self.cached_frame_shape = frame_shape
            self.scale_x = frame_shape[1] / self.imgsz
            self.scale_y = frame_shape[0] / self.imgsz

        box_data[:, (0, 2)] *= self.scale_x
        box_data[:, (1, 3)] *= self.scale_y

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
    if class_id in (0, 1):
        return "c"

    if class_id in (2, 3):
        return "t"

    return None



def find_head_for_body(body, detections, head_class):


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


    aim_region = clamp(aim_region, 0.0, 1.0)

    body, head = get_body_and_head(
        anchor,
        detections,
    )

    
    if body is not None and head is not None:
        body_x, body_y = box_center(body)
        head_x, head_y = box_center(head)

    
    elif body is not None:
        body_x, body_y = box_center(body)

        body_h = box_height(body)

        head_x = body_x
        head_y = body["y1"] + body_h * 0.08

  
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

    heads = [
        detection
        for detection in detections
        if (
            detection["class_id"] in allowed_classes
            and detection["class_id"] in (1, 3)
        )
    ]

    bodies = [
        detection
        for detection in detections
        if (
            detection["class_id"] in allowed_classes
            and detection["class_id"] in (0, 2)
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



def box_iou(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])

    intersection = max(0.0, right - left) * max(
        0.0,
        bottom - top,
    )
    area_a = max(1.0, box_a[2] - box_a[0]) * max(
        1.0,
        box_a[3] - box_a[1],
    )
    area_b = max(1.0, box_b[2] - box_b[0]) * max(
        1.0,
        box_b[3] - box_b[1],
    )
    union = area_a + area_b - intersection
    return intersection / max(1.0, union)


class StableTargetLock:


    MAX_COAST_SECONDS = 0.035
    MAX_HOLD_SECONDS = 0.140

    def __init__(self):
        self.locked = False
        self.aiming_previous = False
        self.last_aim_x = 0.0
        self.last_aim_y = 0.0
        self.last_bbox = None
        self.last_scale = 30.0
        self.last_capture_time = 0.0
        self.last_seen_time = 0.0
        self.last_source = None
        self.last_target = None
        self.missing_frames = 0

    def reset(self):
        self.locked = False
        self.last_aim_x = 0.0
        self.last_aim_y = 0.0
        self.last_bbox = None
        self.last_scale = 30.0
        self.last_capture_time = 0.0
        self.last_seen_time = 0.0
        self.last_source = None
        self.last_target = None
        self.missing_frames = 0

    @staticmethod
    def _virtual_body_from_head(head):
 
        head_x, head_y = box_center(head)
        head_w = box_width(head)
        head_h = box_height(head)
        body_h = max(4.4 * head_h, 2.4 * head_w)
        body_w = max(2.0 * head_w, body_h * 0.34)
        body_top = head_y - head_h * 0.23

        return (
            head_x - body_w * 0.5,
            body_top,
            head_x + body_w * 0.5,
            body_top + body_h,
        )

    @staticmethod
    def _candidate_scale(bbox):
        width = max(1.0, bbox[2] - bbox[0])
        height = max(1.0, bbox[3] - bbox[1])
        return math.hypot(width, height)

    def _build_candidates(
        self,
        detections,
        allowed_classes,
        aim_region,
    ):
        allowed = set(allowed_classes)
        bodies = [
            detection
            for detection in detections
            if (
                detection["class_id"] in allowed
                and detection["class_id"] in (0, 2)
            )
        ]
        heads = [
            detection
            for detection in detections
            if (
                detection["class_id"] in allowed
                and detection["class_id"] in (1, 3)
            )
        ]

        candidates = []
        matched_head_ids = set()

        for body in bodies:
            team = get_team_from_class(body["class_id"])
            head_class = TEAM_PART_CLASSES[team]["head"]
            head = find_head_for_body(
                body,
                detections,
                head_class,
            )

            if head is not None:
                matched_head_ids.add(id(head))

            bbox = (
                body["x1"],
                body["y1"],
                body["x2"],
                body["y2"],
            )
            aim_x, aim_y = get_aim_point(
                body,
                detections,
                aim_region,
            )
            candidates.append({
                "anchor": body,
                "bbox": bbox,
                "measured_aim": (aim_x, aim_y),
                "source": "body",
                "head_measured": head is not None,
                "confidence": max(
                    body["confidence"],
                    0.0 if head is None else head["confidence"],
                ),
                "scale": self._candidate_scale(bbox),
            })


        for head in heads:
            if id(head) in matched_head_ids:
                continue

            bbox = self._virtual_body_from_head(head)
            aim_x, aim_y = get_aim_point(
                head,
                detections,
                aim_region,
            )
            candidates.append({
                "anchor": head,
                "bbox": bbox,
                "measured_aim": (aim_x, aim_y),
                "source": "head",
                "head_measured": True,
                "confidence": head["confidence"],
                "scale": self._candidate_scale(bbox),
            })

        return candidates

    @staticmethod
    def _nearest_candidate(
        candidates,
        center_x,
        center_y,
    ):
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda candidate: (
                (
                    candidate["measured_aim"][0] - center_x
                ) ** 2
                + (
                    candidate["measured_aim"][1] - center_y
                ) ** 2
            ),
        )

    def _acquire(self, candidate, capture_time):
        if candidate is None:
            self.reset()
            return None

        aim_x, aim_y = candidate["measured_aim"]

        self.locked = True
        self.last_aim_x = aim_x
        self.last_aim_y = aim_y
        self.last_bbox = candidate["bbox"]
        self.last_scale = candidate["scale"]
        self.last_capture_time = capture_time
        self.last_seen_time = capture_time
        self.last_source = candidate["source"]
        self.last_target = candidate["anchor"]
        self.missing_frames = 0

        return {
            "target": candidate["anchor"],
            "aim_point": (aim_x, aim_y),
            "target_scale": candidate["scale"],
            "measurement_valid": True,
            "locked": True,
        }

    def _associate(
        self,
        candidates,
        center_x,
        center_y,
        capture_time,
        aim_controller,
    ):
        predicted_x, predicted_y = (
            aim_controller.project_screen_point(
                self.last_aim_x,
                self.last_aim_y,
                self.last_capture_time,
                capture_time,
            )
        )
        shift_x = predicted_x - self.last_aim_x
        shift_y = predicted_y - self.last_aim_y
        predicted_bbox = (
            self.last_bbox[0] + shift_x,
            self.last_bbox[1] + shift_y,
            self.last_bbox[2] + shift_x,
            self.last_bbox[3] + shift_y,
        )

        dt = clamp(
            capture_time - self.last_capture_time,
            0.0,
            0.080,
        )
        speed = math.hypot(
            aim_controller.velocity_x,
            aim_controller.velocity_y,
        )
        gate = (
            max(30.0, self.last_scale * 1.45)
            + min(30.0, speed * dt * 0.60)
        )

        best = None
        best_aim = None
        best_score = float("inf")

        for candidate in candidates:

            aim_x, aim_y = candidate["measured_aim"]
            distance = math.hypot(
                aim_x - predicted_x,
                aim_y - predicted_y,
            )

            scale_ratio = max(
                candidate["scale"] / max(1.0, self.last_scale),
                self.last_scale / max(1.0, candidate["scale"]),
            )

            if distance > gate or scale_ratio > 2.20:
                continue

            overlap = box_iou(
                predicted_bbox,
                candidate["bbox"],
            )
            source_penalty = (
                0.08
                if candidate["source"] != self.last_source
                else 0.0
            )
            score = (
                distance / gate
                + (1.0 - overlap) * 0.38
                + abs(math.log(scale_ratio)) * 0.22
                + source_penalty
                - candidate["confidence"] * 0.04
            )

            if score < best_score:
                best = candidate
                best_aim = (aim_x, aim_y)
                best_score = score

        previous_dx = predicted_x - center_x
        previous_dy = predicted_y - center_y
        previous_radius = math.hypot(
            previous_dx,
            previous_dy,
        )
        zoom_best = None
        zoom_best_aim = None
        zoom_best_score = float("inf")

        for candidate in candidates:
            signed_scale_ratio = (
                candidate["scale"]
                / max(1.0, self.last_scale)
            )

            if not (
                1.30 <= signed_scale_ratio <= 7.00
                or 0.14 <= signed_scale_ratio <= 0.77
            ):
                continue

            aim_x, aim_y = candidate["measured_aim"]
            expected_x = (
                center_x + previous_dx * signed_scale_ratio
            )
            expected_y = (
                center_y + previous_dy * signed_scale_ratio
            )
            expected_distance = math.hypot(
                aim_x - expected_x,
                aim_y - expected_y,
            )
            zoom_gate = max(
                36.0,
                candidate["scale"] * 0.68,
                self.last_scale * 1.45,
            )

            if expected_distance > zoom_gate:
                continue

            current_dx = aim_x - center_x
            current_dy = aim_y - center_y
            current_radius = math.hypot(
                current_dx,
                current_dy,
            )
            direction_penalty = 0.0

            if (
                previous_radius > 10.0
                and current_radius > 10.0
            ):
                cosine = (
                    previous_dx * current_dx
                    + previous_dy * current_dy
                ) / (
                    previous_radius * current_radius
                )

                if cosine < 0.10:
                    continue

                direction_penalty = (
                    1.0 - cosine
                ) * 0.35

            source_penalty = (
                0.06
                if candidate["source"] != self.last_source
                else 0.0
            )
            score = (
                expected_distance / zoom_gate
                + direction_penalty
                + source_penalty
                - candidate["confidence"] * 0.05
            )

            if score < zoom_best_score:
                zoom_best = candidate
                zoom_best_aim = (aim_x, aim_y)
                zoom_best_score = score

        if zoom_best is not None:
            return zoom_best, zoom_best_aim, True

        if best is not None:
            signed_scale_ratio = (
                best["scale"] / max(1.0, self.last_scale)
            )
            view_scale_changed = (
                signed_scale_ratio >= 1.35
                or signed_scale_ratio <= 0.74
            )
            return best, best_aim, view_scale_changed

        return None, (predicted_x, predicted_y), False

    def update(
        self,
        detections,
        allowed_classes,
        center_x,
        center_y,
        aim_region,
        aiming,
        capture_time,
        aim_controller,
    ):
        candidates = self._build_candidates(
            detections,
            allowed_classes,
            aim_region,
        )

        if not aiming:
            self.reset()
            self.aiming_previous = False
            preview = self._nearest_candidate(
                candidates,
                center_x,
                center_y,
            )

            if preview is None:
                return {
                    "target": None,
                    "aim_point": None,
                    "target_scale": 30.0,
                    "measurement_valid": False,
                    "locked": False,
                }

            return {
                "target": preview["anchor"],
                "aim_point": preview["measured_aim"],
                "target_scale": preview["scale"],
                "measurement_valid": True,
                "locked": False,
            }

        if not self.aiming_previous or not self.locked:
            self.aiming_previous = True
            return self._acquire(
                self._nearest_candidate(
                    candidates,
                    center_x,
                    center_y,
                ),
                capture_time,
            ) or {
                "target": None,
                "aim_point": None,
                "target_scale": 30.0,
                "measurement_valid": False,
                "locked": False,
            }

        candidate, aim_point, view_scale_changed = self._associate(
            candidates,
            center_x,
            center_y,
            capture_time,
            aim_controller,
        )

        if candidate is not None:
            if view_scale_changed:
                
                aim_controller.reset()

            self.last_aim_x, self.last_aim_y = aim_point
            self.last_bbox = candidate["bbox"]
            self.last_scale = candidate["scale"]
            self.last_capture_time = capture_time
            self.last_seen_time = capture_time
            self.last_source = candidate["source"]
            self.last_target = candidate["anchor"]
            self.missing_frames = 0

            return {
                "target": candidate["anchor"],
                "aim_point": aim_point,
                "target_scale": candidate["scale"],
                "measurement_valid": True,
                "locked": True,
            }

        self.missing_frames += 1
        missing_time = capture_time - self.last_seen_time

        if (
            missing_time <= self.MAX_COAST_SECONDS
            and self.missing_frames <= 2
        ):
            self.last_aim_x, self.last_aim_y = aim_point
            self.last_capture_time = capture_time
            return {
                "target": self.last_target,
                "aim_point": aim_point,
                "target_scale": self.last_scale,
                "measurement_valid": False,
                "locked": True,
            }

        if missing_time > self.MAX_HOLD_SECONDS:
            self.reset()
            self.aiming_previous = True

        return {
            "target": None,
            "aim_point": None,
            "target_scale": self.last_scale,
            "measurement_valid": False,
            "locked": self.locked,
        }


# ============================================================
# 调试画面
# ============================================================

def draw_debug(
    frame,
    detections,
    target,
    tracked_aim_point,
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

    if tracked_aim_point is not None:
        aim_x, aim_y = tracked_aim_point
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
        "t/th (2/3)"
        if target_mode == "t"
        else "c/ch (0/1)"
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
        "F1:t/th F2:c/ch",
        "F3/F4:X  F5/F6:Y  F7/F8:Smooth",
        "[/]:Aim region  F9/F10:ROI",
        "F11:Trigger  F12:Debug  END:Quit",
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


    cv2.setNumThreads(1)

    try:
        cv2.ocl.setUseOpenCL(False)
    except AttributeError:
        pass

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

    model = YOLO(str(model_path), task="detect")

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

        # 固定256×256输入允许cuDNN在启动预热阶段比较全部可用卷积
        # 方案并缓存最快结果。只增加一次启动时间，不增加逐帧开销。
        if hasattr(
            torch.backends.cudnn,
            "benchmark_limit",
        ):
            torch.backends.cudnn.benchmark_limit = 0

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    # fuse() 仅适用于 PyTorch .pt 模型。TensorRT/ONNX 等导出模型
    # 已在导出阶段完成图优化，只支持 predict/val，不能再次融合。
    if model_path.suffix.lower() == ".pt":
        try:
            model.fuse()
        except (AttributeError, RuntimeError, TypeError):
            # 兼容已融合或不支持 fuse 的特殊 PyTorch 模型。
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
        1.0,
    )

    aim_region = clamp(
        args.aim_region,
        0.0,
        1.0,
    )

    # 默认锁定 t/th（2/3）
    target_mode = "t"
    target_classes = (2, 3)

    debug_enabled = not args.no_debug
    debug_window_created = False
    auto_trigger_enabled = False
    trigger_range = DEFAULT_TRIGGER_RANGE
    trigger_interval_ms = DEFAULT_TRIGGER_INTERVAL_MS
    trigger_in_range = False

    hotkeys = HotkeyState()
    aim_controller = AimController()
    auto_trigger_controller = AutoTriggerController()
    target_lock = StableTargetLock()
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
    print("F1：锁定 t/th（类别2和3）")
    print("F2：锁定 c/ch（类别0和1）")
    print("F3/F4：降低/提高X轴强度")
    print("F5/F6：降低/提高Y轴强度")
    print("F7/F8：降低/提高平滑强度")
    print("[ / ]：降低/提高锁点区域")
    print("F9/F10：缩小/扩大检测区域")
    print("F11：开启/关闭自动点击")
    print(': / "：减小/增大自动点击范围（0～10像素）')
    print(
        "< / >：减小/增大自动点击间隔"
        "（10～100毫秒，默认50毫秒）"
    )
    print("F12：显示/关闭调试窗口")
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
            "classes": (2, 3),
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

            hotkeys.begin_frame()

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
                target_mode = "t"
                target_classes = (2, 3)

                aim_controller.reset()
                target_lock.reset()

                print("锁定类别：t/th（2和3）")

            if hotkeys.pressed_once(VK_F2):
                target_mode = "c"
                target_classes = (0, 1)

                aim_controller.reset()
                target_lock.reset()

                print("锁定类别：c/ch（0和1）")

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
                target_lock.reset()

                print(
                    f"平滑强度：{smoothing:.2f}"
                )

            if hotkeys.pressed_once(VK_F8):
                smoothing = min(
                    1.0,
                    smoothing + 0.05,
                )

                aim_controller.reset()
                target_lock.reset()

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
                target_lock.reset()

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
                target_lock.reset()

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
                target_lock.reset()

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
                target_lock.reset()

                print(
                    f"检测区域：{roi_size}×{roi_size}"
                )

            if hotkeys.pressed_once(VK_F11):
                auto_trigger_enabled = (
                    not auto_trigger_enabled
                )
                auto_trigger_controller.reset()

                print(
                    "自动点击："
                    + (
                        "开启"
                        if auto_trigger_enabled
                        else "关闭"
                    )
                )

            if hotkeys.pressed_once(VK_OEM_1):
                trigger_range = max(
                    0,
                    trigger_range - 1,
                )
                auto_trigger_controller.reset()

                print(
                    f"自动点击范围：{trigger_range}像素"
                )

            if hotkeys.pressed_once(VK_OEM_7):
                trigger_range = min(
                    10,
                    trigger_range + 1,
                )
                auto_trigger_controller.reset()

                print(
                    f"自动点击范围：{trigger_range}像素"
                )

            if hotkeys.pressed_once(VK_OEM_COMMA):
                trigger_interval_ms = max(
                    10,
                    trigger_interval_ms
                    - TRIGGER_INTERVAL_STEP_MS,
                )
                auto_trigger_controller.reset()

                print(
                    "自动点击单击间隔："
                    f"{trigger_interval_ms}毫秒"
                )

            if hotkeys.pressed_once(VK_OEM_PERIOD):
                trigger_interval_ms = min(
                    100,
                    trigger_interval_ms
                    + TRIGGER_INTERVAL_STEP_MS,
                )
                auto_trigger_controller.reset()

                print(
                    "自动点击单击间隔："
                    f"{trigger_interval_ms}毫秒"
                )

            if hotkeys.pressed_once(VK_F12):
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

            frame, capture_time = screen_capture.get_frame()

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
                predict_arguments["classes"] = target_classes

                results = model.predict(
                    **predict_arguments
                )

                detections = extract_detections(
                    results[0]
                )

            frame_height, frame_width = frame.shape[:2]

            center_x = frame_width / 2.0
            center_y = frame_height / 2.0


            aiming = is_key_down(VK_RBUTTON)

            lock_state = target_lock.update(
                detections=detections,
                allowed_classes=target_classes,
                center_x=center_x,
                center_y=center_y,
                aim_region=aim_region,
                aiming=aiming,
                capture_time=capture_time,
                aim_controller=aim_controller,
            )
            target = lock_state["target"]
            tracked_aim_point = lock_state["aim_point"]

            # ------------------------------------------------
            # 计算并发送鼠标移动
            # ------------------------------------------------

            if (
                aiming
                and tracked_aim_point is not None
            ):
                aim_controller.update(
                    aim_x=tracked_aim_point[0],
                    aim_y=tracked_aim_point[1],
                    center_x=center_x,
                    center_y=center_y,
                    capture_time=capture_time,
                    target_scale=lock_state["target_scale"],
                    x_strength=x_strength,
                    y_strength=y_strength,
                    smoothing=smoothing,
                    max_step=args.max_step,
                    deadzone=args.deadzone,
                    invert_y=args.invert_y,
                    measurement_valid=lock_state[
                        "measurement_valid"
                    ],
                )

            else:
                aim_controller.reset()

            # ------------------------------------------------
            # 自动扳机
            # ------------------------------------------------

            trigger_in_range = auto_trigger_controller.update(
                enabled=(
                    auto_trigger_enabled
                    and aiming
                    and lock_state["locked"]
                ),
                aim_point=tracked_aim_point,
                center_x=center_x,
                center_y=center_y,
                trigger_range=trigger_range,
                trigger_interval_ms=trigger_interval_ms,
                measurement_valid=lock_state[
                    "measurement_valid"
                ],
            )

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
                    auto_trigger_enabled=(
                        auto_trigger_enabled
                    ),
                    trigger_range=trigger_range,
                    trigger_interval_ms=(
                        trigger_interval_ms
                    ),
                    trigger_in_range=trigger_in_range,
                    debug_enabled=debug_enabled,
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
                        tracked_aim_point=tracked_aim_point,
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

                # 窗口消息最高按30 Hz处理；识别循环达到120 FPS时无需
                # 额外执行120次GUI调用。pollKey本身不主动等待。
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

            elif parameter_overlay_enabled:
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