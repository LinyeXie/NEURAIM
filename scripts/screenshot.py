from __future__ import annotations

import ctypes
import json
import os
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import (
    BooleanVar,
    Checkbutton,
    Frame,
    Label,
    StringVar,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
)
from tkinter import ttk
from ctypes import wintypes

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None


APP_NAME = "中央区域截图工具"
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 800
MIN_SIZE = 32
HOTKEY_ID_MAIN_PLUS = 0x5101
HOTKEY_ID_NUMPAD_PLUS = 0x5102
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_SHIFT = 0x0004
MOD_NOREPEAT = 0x4000
VK_OEM_PLUS = 0xBB
VK_ADD = 0x6B


def enable_dpi_awareness() -> None:
    """Keep screen coordinates accurate when Windows display scaling is enabled."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def default_output_dir() -> Path:
    pictures = Path(r"E:\Floder\VsCodeFile\PyCodes\yolo_enemy_detector\data\screenshot")
    base = pictures if pictures.exists() else Path.home()
    return base / "Screenshots"


def config_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home()))
    return base / "CenterScreenshot" / "settings.json"


def load_config() -> dict[str, object]:
    defaults: dict[str, object] = {
        "output_dir": str(default_output_dir()),
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "format": "PNG",
        "beep": True,
    }
    path = config_path()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            defaults.update(loaded)
    except (OSError, ValueError, TypeError):
        pass
    return defaults


def save_config(config: dict[str, object]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


class WindowsHotkeyThread(threading.Thread):
    """Register both common '+' keys and forward events to the Tk main thread."""

    def __init__(self, events: queue.Queue[tuple[str, object]]) -> None:
        super().__init__(name="global-plus-hotkey", daemon=True)
        self.events = events
        self.thread_id = 0

    def run(self) -> None:
        user32 = ctypes.windll.user32
        self.thread_id = int(ctypes.windll.kernel32.GetCurrentThreadId())
        registered: list[int] = []

        hotkeys = (
            (HOTKEY_ID_MAIN_PLUS, MOD_SHIFT | MOD_NOREPEAT, VK_OEM_PLUS),
            (HOTKEY_ID_NUMPAD_PLUS, MOD_NOREPEAT, VK_ADD),
        )

        try:
            for hotkey_id, modifiers, virtual_key in hotkeys:
                if user32.RegisterHotKey(
                    None, hotkey_id, modifiers, virtual_key
                ):
                    registered.append(hotkey_id)

            self.events.put(("hotkey_ready", tuple(registered)))

            if not registered:
                return

            message = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                if (
                    message.message == WM_HOTKEY
                    and int(message.wParam) in registered
                ):
                    self.events.put(("capture", None))
        finally:
            for hotkey_id in registered:
                user32.UnregisterHotKey(None, hotkey_id)

    def stop(self) -> None:
        if self.thread_id:
            ctypes.windll.user32.PostThreadMessageW(
                self.thread_id, WM_QUIT, 0, 0
            )


class ScreenshotApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.hotkey_thread: WindowsHotkeyThread | None = None
        self.capture_lock = threading.Lock()
        self.preview_window: Toplevel | None = None
        self.closing = False

        settings = load_config()
        self.output_var = StringVar(
            value=str(settings.get("output_dir", default_output_dir()))
        )
        self.width_var = StringVar(value=str(settings.get("width", DEFAULT_WIDTH)))
        self.height_var = StringVar(
            value=str(settings.get("height", DEFAULT_HEIGHT))
        )
        self.format_var = StringVar(value=str(settings.get("format", "PNG")).upper())
        self.beep_var = BooleanVar(value=bool(settings.get("beep", True)))
        self.status_var = StringVar(value="正在注册全局 + 热键……")

        self._build_ui()
        self._place_window()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self._process_events)
        self._start_hotkeys()

    def _build_ui(self) -> None:
        self.root.title(APP_NAME)
        self.root.resizable(False, False)

        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(1, weight=1)

        title = ttk.Label(
            outer,
            text="中央区域截图",
            font=("Microsoft YaHei UI", 16, "bold"),
        )
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        subtitle = ttk.Label(
            outer,
            text="最小化程序后，按主键盘 Shift+= 或数字小键盘 + 自动保存。",
            foreground="#555555",
        )
        subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 16))

        ttk.Label(outer, text="保存文件夹").grid(
            row=2, column=0, sticky="w", padx=(0, 10), pady=5
        )
        output_entry = ttk.Entry(outer, textvariable=self.output_var, width=48)
        output_entry.grid(row=2, column=1, sticky="ew", pady=5)
        ttk.Button(outer, text="浏览…", command=self.choose_folder).grid(
            row=2, column=2, padx=(8, 0), pady=5
        )

        ttk.Label(outer, text="截图宽度").grid(
            row=3, column=0, sticky="w", padx=(0, 10), pady=5
        )
        width_spin = ttk.Spinbox(
            outer,
            from_=MIN_SIZE,
            to=16384,
            increment=16,
            textvariable=self.width_var,
            width=12,
        )
        width_spin.grid(row=3, column=1, sticky="w", pady=5)
        ttk.Label(outer, text="像素").grid(row=3, column=1, sticky="w", padx=(98, 0))

        ttk.Label(outer, text="截图高度").grid(
            row=4, column=0, sticky="w", padx=(0, 10), pady=5
        )
        height_spin = ttk.Spinbox(
            outer,
            from_=MIN_SIZE,
            to=16384,
            increment=16,
            textvariable=self.height_var,
            width=12,
        )
        height_spin.grid(row=4, column=1, sticky="w", pady=5)
        ttk.Label(outer, text="像素").grid(row=4, column=1, sticky="w", padx=(98, 0))

        ttk.Label(outer, text="图片格式").grid(
            row=5, column=0, sticky="w", padx=(0, 10), pady=5
        )
        format_menu = ttk.OptionMenu(
            outer,
            self.format_var,
            self.format_var.get(),
            "PNG",
            "JPG",
            "BMP",
        )
        format_menu.grid(row=5, column=1, sticky="w", pady=5)

        Checkbutton(
            outer,
            text="保存成功后播放提示音",
            variable=self.beep_var,
        ).grid(row=6, column=1, sticky="w", pady=(4, 8))

        separator = ttk.Separator(outer, orient="horizontal")
        separator.grid(row=7, column=0, columnspan=3, sticky="ew", pady=8)

        controls = ttk.Frame(outer)
        controls.grid(row=8, column=0, columnspan=3, sticky="ew")
        ttk.Button(
            controls,
            text="立即截图",
            command=lambda: self.request_capture("按钮"),
        ).pack(side="left")
        ttk.Button(
            controls,
            text="预览截图区域",
            command=self.preview_region,
        ).pack(side="left", padx=8)
        ttk.Button(
            controls,
            text="最小化",
            command=self.root.iconify,
        ).pack(side="right")

        status_frame = ttk.Frame(outer)
        status_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        self.status_dot = Label(
            status_frame,
            text="●",
            fg="#d28b00",
            font=("Segoe UI", 11),
        )
        self.status_dot.pack(side="left")
        ttk.Label(status_frame, textvariable=self.status_var).pack(
            side="left", padx=(6, 0)
        )

    def _place_window(self) -> None:
        self.root.update_idletasks()
        width = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()
        screen_width = self.root.winfo_screenwidth()
        x = max(10, screen_width - width - 30)
        self.root.geometry(f"{width}x{height}+{x}+30")

    def _start_hotkeys(self) -> None:
        if sys.platform != "win32":
            self._set_status("此程序仅支持 Windows 全局热键。", "error")
            return
        self.hotkey_thread = WindowsHotkeyThread(self.events)
        self.hotkey_thread.start()

    def _process_events(self) -> None:
        if self.closing:
            return

        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "capture":
                    self.request_capture("+ 热键")
                elif event == "hotkey_ready":
                    registered = tuple(payload) if payload else ()
                    if len(registered) == 2:
                        self._set_status("全局 + 热键已启用，可以开始截图。", "ready")
                    elif registered:
                        self._set_status(
                            "仅注册了一个 + 热键；另一个可能被其他软件占用。",
                            "warning",
                        )
                    else:
                        self._set_status(
                            "+ 热键注册失败，可能已被其他软件占用。",
                            "error",
                        )
                elif event == "capture_done":
                    self.capture_lock.release()
                    path = Path(str(payload))
                    self._set_status(f"已保存：{path.name}", "ready")
                    if self.beep_var.get() and sys.platform == "win32":
                        try:
                            import winsound

                            winsound.MessageBeep(winsound.MB_OK)
                        except Exception:
                            pass
                elif event == "capture_error":
                    self.capture_lock.release()
                    self._set_status(f"截图失败：{payload}", "error")
        except queue.Empty:
            pass

        self.root.after(30, self._process_events)

    def _set_status(self, message: str, state: str) -> None:
        colors = {
            "ready": "#16833a",
            "working": "#2474cc",
            "warning": "#d28b00",
            "error": "#c62f2f",
        }
        self.status_var.set(message)
        self.status_dot.configure(fg=colors.get(state, "#666666"))

    def choose_folder(self) -> None:
        initial = Path(self.output_var.get()).expanduser()
        chosen = filedialog.askdirectory(
            title="选择截图保存文件夹",
            initialdir=str(initial if initial.exists() else Path.home()),
        )
        if chosen:
            self.output_var.set(chosen)
            self._save_current_config()

    def _read_dimensions(self) -> tuple[int, int]:
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
        except ValueError as exc:
            raise ValueError("宽度和高度必须是整数") from exc

        if width < MIN_SIZE or height < MIN_SIZE:
            raise ValueError(f"宽度和高度不能小于 {MIN_SIZE} 像素")
        return width, height

    def _screen_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        user32 = ctypes.windll.user32
        screen_width = int(user32.GetSystemMetrics(0))
        screen_height = int(user32.GetSystemMetrics(1))

        width = min(width, screen_width)
        height = min(height, screen_height)
        left = (screen_width - width) // 2
        top = (screen_height - height) // 2
        return left, top, left + width, top + height

    def request_capture(self, source: str) -> None:
        self._hide_preview()

        if ImageGrab is None:
            messagebox.showerror(
                "缺少 Pillow",
                "请先在命令行执行：\n\npip install pillow",
            )
            self._set_status("缺少 Pillow，无法截图。", "error")
            return

        if not self.capture_lock.acquire(blocking=False):
            self._set_status("上一张截图仍在保存，请稍候。", "warning")
            return

        try:
            width, height = self._read_dimensions()
            output_dir = Path(self.output_var.get()).expanduser()
            if not str(output_dir).strip():
                raise ValueError("请先选择保存文件夹")
            image_format = self.format_var.get().upper()
            if image_format not in {"PNG", "JPG", "BMP"}:
                raise ValueError("图片格式只能是 PNG、JPG 或 BMP")

            output_dir.mkdir(parents=True, exist_ok=True)
            box = self._screen_box(width, height)
            self._save_current_config()
        except (OSError, ValueError) as exc:
            self.capture_lock.release()
            self._set_status(str(exc), "error")
            messagebox.showerror("设置错误", str(exc))
            return

        actual_width = box[2] - box[0]
        actual_height = box[3] - box[1]
        self._set_status(
            f"{source}：正在截取中央 {actual_width}×{actual_height}…",
            "working",
        )

        worker = threading.Thread(
            target=self._capture_worker,
            args=(output_dir, image_format, box),
            name="screenshot-save",
            daemon=True,
        )
        worker.start()

    def _capture_worker(
        self,
        output_dir: Path,
        image_format: str,
        box: tuple[int, int, int, int],
    ) -> None:
        try:
            image = ImageGrab.grab(bbox=box)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            extension = image_format.lower()
            path = output_dir / f"screenshot_{timestamp}.{extension}"

            counter = 1
            while path.exists():
                path = output_dir / (
                    f"screenshot_{timestamp}_{counter:02d}.{extension}"
                )
                counter += 1

            if image_format == "JPG":
                image = image.convert("RGB")
                image.save(path, format="JPEG", quality=95, subsampling=0)
            else:
                image.save(path, format=image_format)

            self.events.put(("capture_done", str(path)))
        except Exception as exc:
            self.events.put(("capture_error", str(exc)))

    def preview_region(self) -> None:
        try:
            width, height = self._read_dimensions()
            left, top, right, bottom = self._screen_box(width, height)
        except ValueError as exc:
            messagebox.showerror("设置错误", str(exc))
            return

        if self.preview_window is not None:
            try:
                self.preview_window.destroy()
            except Exception:
                pass

        preview = Toplevel(self.root)
        self.preview_window = preview
        preview.overrideredirect(True)
        preview.attributes("-topmost", True)
        preview.attributes("-alpha", 0.25)
        preview.configure(bg="#00a8ff")
        preview.geometry(
            f"{right - left}x{bottom - top}+{left}+{top}"
        )

        border = Frame(
            preview,
            bg="#00a8ff",
            highlightbackground="#0078d4",
            highlightthickness=5,
        )
        border.pack(fill="both", expand=True)
        Label(
            border,
            text=f"中央截图区域  {right - left} × {bottom - top}",
            bg="#0078d4",
            fg="white",
            font=("Microsoft YaHei UI", 14, "bold"),
            padx=12,
            pady=7,
        ).place(relx=0.5, rely=0.5, anchor="center")

        preview.after(1300, self._hide_preview)

    def _hide_preview(self) -> None:
        if self.preview_window is not None:
            try:
                self.preview_window.destroy()
            except Exception:
                pass
            self.preview_window = None

    def _save_current_config(self) -> None:
        try:
            width, height = self._read_dimensions()
            save_config(
                {
                    "output_dir": self.output_var.get(),
                    "width": width,
                    "height": height,
                    "format": self.format_var.get().upper(),
                    "beep": self.beep_var.get(),
                }
            )
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        self.closing = True
        self._save_current_config()
        self._hide_preview()
        if self.hotkey_thread is not None:
            self.hotkey_thread.stop()
        self.root.destroy()


def main() -> int:
    if sys.platform != "win32":
        print("ERROR: 此截图工具目前仅支持 Windows。", file=sys.stderr)
        return 1

    if ImageGrab is None:
        print(
            "ERROR: 缺少 Pillow。请执行：pip install pillow",
            file=sys.stderr,
        )
        return 1

    enable_dpi_awareness()
    root = Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    ScreenshotApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())