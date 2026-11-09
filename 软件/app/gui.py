# -*- coding: utf-8 -*-
"""Tkinter 图形界面。

设计目标：
  * 摄像头打不开、没装驱动、画面全黑时，程序仍然可用（可打开图片分析）；
  * 所有异常都被捕获并写入日志，界面只显示友好提示；
  * 不依赖 PIL / PyQt 等额外库，PyInstaller 打包体积小、兼容性好。
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np

from .config import APP_NAME, DISCLAIMER_SHORT, VERSION, reports_dir
from .demo import PALETTES, make_tongue_image
from .features import FeatureError, extract_features
from .imaging import (
    ImageError,
    Segmentation,
    TongueNotFoundError,
    annotate,
    assess_capture_quality,
    default_roi,
    load_image,
    resize_max,
    segment_tongue,
)
from .knowledge import KnowledgeBaseError, evaluate, load_knowledge_base
from .report import Report, build_report
from .utils import setup_logging

log = logging.getLogger(__name__)

CANVAS_W, CANVAS_H = 640, 480
MIN_ROI = 48


def _font_family() -> str:
    if sys.platform.startswith("win"):
        return "Microsoft YaHei UI"
    if sys.platform == "darwin":
        return "PingFang SC"
    return "Noto Sans CJK SC"


def _open_capture(index: int):
    """按平台依次尝试可用的摄像头后端。"""
    backends = []
    try:
        if sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif sys.platform == "darwin":
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    except Exception:
        backends = [cv2.CAP_ANY]

    seen = set()
    for backend in backends:
        if backend in seen:
            continue
        seen.add(backend)
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                # 预热，丢掉前几帧（部分摄像头首帧很暗）
                for _ in range(4):
                    cap.read()
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return None


class CameraWorker(threading.Thread):
    """后台读帧线程，避免界面卡顿。"""

    def __init__(self, index: int = 0):
        super().__init__(daemon=True, name="camera-worker")
        self.index = index
        self.frame: Optional[np.ndarray] = None
        self.error: Optional[str] = None
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = None

    def run(self) -> None:
        try:
            cap = _open_capture(self.index)
        except Exception as exc:  # noqa: BLE001
            self.error = f"摄像头初始化失败：{exc}"
            return
        if cap is None:
            self.error = (
                "未能打开摄像头。请检查：\n"
                "· 摄像头是否被其他软件占用\n"
                "· Windows 设置 → 隐私 → 摄像头 是否允许桌面应用访问\n"
                "· 是否插好 USB 摄像头\n"
                "也可以点击【打开图片】分析已有照片。"
            )
            return
        self._cap = cap
        fails = 0
        while not self._stop.is_set():
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None or frame.size == 0:
                fails += 1
                if fails >= 40:
                    self.error = "摄像头持续读取失败，请重新连接后点击【打开摄像头】。"
                    break
                time.sleep(0.05)
                continue
            fails = 0
            with self.lock:
                self.frame = frame
            time.sleep(0.01)
        try:
            cap.release()
        except Exception:
            pass

    def snapshot(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self._stop.set()


class TongueApp:
    def __init__(
        self,
        root: tk.Tk,
        kb: dict,
        demo: Optional[str] = None,
        camera_index: int = 0,
    ):
        self.root = root
        self.kb = kb
        self.camera_index = camera_index
        self.worker: Optional[CameraWorker] = None
        self.current_frame: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.roi: Optional[tuple] = None
        self.mode = "idle"
        self.last_report: Optional[Report] = None
        self.last_paths: dict = {}
        self._alive = True
        self._photo = None
        self._disp = (0, 0, 1.0)  # offset_x, offset_y, scale
        self._camera_error_shown = False
        self._drag_start = None
        self._rubber = None
        self._render_job = None
        self._is_demo = bool(demo)

        self.family = _font_family()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._set_status("就绪。点击【打开摄像头】或【打开图片】开始。")

        if demo:
            try:
                name = demo if demo in PALETTES else "pale_scallop"
                img = make_tongue_image(name)
                self._load_array(
                    img, source=f"内置示例图（合成）：{name}", is_demo=True
                )
                self.root.after(300, self.analyze_current)
            except Exception:
                log.exception("demo 加载失败")

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.root.title(f"{APP_NAME} v{VERSION}")
        self.root.geometry("1200x780")
        self.root.minsize(1000, 660)
        try:
            self.root.configure(bg="#f5f6f7")
        except Exception:
            pass

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", font=(self.family, 10), padding=(10, 6))
        style.configure("TLabel", font=(self.family, 10))
        style.configure("Hint.TLabel", foreground="#555")

        # 顶部标题与免责声明
        top = ttk.Frame(self.root, padding=(14, 10, 14, 4))
        top.pack(fill="x")
        ttk.Label(
            top,
            text="舌象观察与健康参考",
            font=(self.family, 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            top,
            text=DISCLAIMER_SHORT,
            style="Hint.TLabel",
            wraplength=1100,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))

        # 主体：左图右报告
        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=8)

        left = ttk.Frame(body, padding=4)
        right = ttk.Frame(body, padding=4)
        body.add(left, weight=3)
        body.add(right, weight=4)

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_W,
            height=CANVAS_H,
            bg="#111",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        ttk.Label(
            left,
            text="拍摄要点：正对光源、张大嘴伸舌，让舌体填满中央方框；"
            "可按住鼠标左键拖动重新框选舌体。",
            style="Hint.TLabel",
            wraplength=620,
            justify="left",
        ).pack(anchor="w", pady=(6, 4))

        btns1 = ttk.Frame(left)
        btns1.pack(fill="x", pady=(2, 2))
        self.btn_cam = ttk.Button(btns1, text="打开摄像头", command=self.start_camera)
        self.btn_cam.pack(side="left", padx=(0, 6))
        self.btn_cam_off = ttk.Button(
            btns1, text="关闭摄像头", command=self.stop_camera
        )
        self.btn_cam_off.pack(side="left", padx=(0, 6))
        ttk.Button(btns1, text="拍照并分析 (空格)", command=self.analyze_current).pack(
            side="left", padx=(0, 6)
        )

        btns2 = ttk.Frame(left)
        btns2.pack(fill="x", pady=(2, 2))
        ttk.Button(btns2, text="打开图片", command=self.open_image).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btns2, text="重新框选", command=self.reset_roi).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btns2, text="保存报告", command=self.save_report).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btns2, text="打印报告", command=self.print_report).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btns2, text="报告目录", command=self.open_reports_dir).pack(
            side="left", padx=(0, 6)
        )

        # 右侧报告
        head = ttk.Frame(right)
        head.pack(fill="x")
        ttk.Label(head, text="观察报告", font=(self.family, 12, "bold")).pack(side="left")
        ttk.Label(
            head,
            text="（规则匹配度 ≠ 患病概率）",
            style="Hint.TLabel",
        ).pack(side="left", padx=(8, 0))

        text_wrap = ttk.Frame(right)
        text_wrap.pack(fill="both", expand=True, pady=(6, 0))
        self.report_text = tk.Text(
            text_wrap,
            wrap="word",
            font=(self.family, 10),
            bg="#ffffff",
            relief="solid",
            borderwidth=1,
            padx=12,
            pady=10,
            spacing1=1,
            spacing3=1,
        )
        sb = ttk.Scrollbar(text_wrap, orient="vertical", command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=sb.set)
        self.report_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.report_text.insert(
            "1.0",
            "报告会显示在这里。\n\n"
            "使用步骤：\n"
            "1) 打开摄像头，或打开一张舌象照片；\n"
            "2) 调整取景框，让舌体尽量填满方框；\n"
            "3) 点击【拍照并分析】；\n"
            "4) 查看报告，可保存或打印。\n\n"
            "提示：本软件不输出疾病名称，只给出舌象征象描述与健康参考。",
        )
        self.report_text.configure(state="disabled")

        # 状态栏
        self.status_var = tk.StringVar(value="")
        status = ttk.Frame(self.root, padding=(14, 4, 14, 8))
        status.pack(fill="x")
        ttk.Label(
            status, textvariable=self.status_var, style="Hint.TLabel"
        ).pack(side="left")
        ttk.Label(
            status,
            text="不联网 · 不上传 · 数据仅存本机",
            style="Hint.TLabel",
        ).pack(side="right")

        # 快捷键
        self.root.bind("<space>", lambda e: self.analyze_current())
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_report())
        self.root.bind("<Control-p>", lambda e: self.print_report())

    # -------------------------------------------------------------- helpers
    def _set_status(self, text: str) -> None:
        try:
            self.status_var.set(text)
        except Exception:
            pass

    def _set_report(self, text: str) -> None:
        try:
            self.report_text.configure(state="normal")
            self.report_text.delete("1.0", "end")
            self.report_text.insert("1.0", text)
            self.report_text.configure(state="disabled")
            self.report_text.yview_moveto(0.0)
        except Exception:
            log.exception("显示报告失败")

    def _on_close(self) -> None:
        self._alive = False
        self.stop_camera()
        try:
            self.root.destroy()
        except Exception:
            pass

    # -------------------------------------------------------------- 显示
    def _on_canvas_resize(self, _event=None) -> None:
        if self._render_job is not None:
            try:
                self.root.after_cancel(self._render_job)
            except Exception:
                pass
        self._render_job = self.root.after(120, self._render_current)

    def _render_current(self) -> None:
        self._render_job = None
        if self.current_frame is not None:
            self._show_frame(self.current_frame)

    def _show_frame(self, bgr: np.ndarray) -> None:
        try:
            cw = max(120, self.canvas.winfo_width())
            ch = max(120, self.canvas.winfo_height())
            h, w = bgr.shape[:2]
            scale = min(cw / float(w), ch / float(h))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            disp = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            photo = self._to_photo(disp)
            ox, oy = (cw - nw) // 2, (ch - nh) // 2
            self._disp = (ox, oy, scale)
            self.canvas.delete("all")
            self.canvas.create_image(ox, oy, anchor="nw", image=photo)
            self._photo = photo  # 防止被垃圾回收
            self._draw_roi()
        except Exception:
            log.exception("画面渲染失败")

    def _to_photo(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        header = ("P6 %d %d 255 " % (w, h)).encode("ascii")
        raw = header + rgb.tobytes()
        try:
            return tk.PhotoImage(data=raw)
        except Exception:
            # 兜底：写成临时 PNG 再读，兼容性最好
            try:
                tmp = Path(tempfile.gettempdir()) / "tongue_preview.png"
                ok, buf = cv2.imencode(".png", bgr)
                if not ok:
                    raise RuntimeError("encode failed")
                tmp.write_bytes(buf.tobytes())
                return tk.PhotoImage(file=str(tmp), format="png")
            except Exception:
                return tk.PhotoImage(width=1, height=1)

    def _draw_roi(self) -> None:
        self.canvas.delete("roi")
        if self.current_frame is None or self.roi is None:
            return
        ox, oy, scale = self._disp
        x, y, w, h = self.roi
        x0, y0 = ox + x * scale, oy + y * scale
        x1, y1 = x0 + w * scale, y0 + h * scale
        self.canvas.create_rectangle(
            x0, y0, x1, y1, outline="#00e5a0", width=2, tags="roi"
        )
        self.canvas.create_text(
            x0 + 4,
            y0 + 4,
            anchor="nw",
            text="舌体取景框",
            fill="#00e5a0",
            font=(self.family, 9),
            tags="roi",
        )

    # -------------------------------------------------------------- 鼠标框选
    def _canvas_to_frame(self, cx: float, cy: float):
        ox, oy, scale = self._disp
        if scale <= 0:
            return 0, 0
        return int((cx - ox) / scale), int((cy - oy) / scale)

    def _on_mouse_down(self, event) -> None:
        if self.current_frame is None:
            return
        self._drag_start = (event.x, event.y)

    def _on_mouse_move(self, event) -> None:
        if self._drag_start is None or self.current_frame is None:
            return
        self.canvas.delete("rubber")
        x0, y0 = self._drag_start
        self._rubber = self.canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#ffd166", width=2, tags="rubber"
        )

    def _on_mouse_up(self, event) -> None:
        if self._drag_start is None or self.current_frame is None:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        self.canvas.delete("rubber")
        if abs(event.x - x0) < 20 or abs(event.y - y0) < 20:
            return
        fx0, fy0 = self._canvas_to_frame(min(x0, event.x), min(y0, event.y))
        fx1, fy1 = self._canvas_to_frame(max(x0, event.x), max(y0, event.y))
        h, w = self.current_frame.shape[:2]
        fx0 = max(0, min(fx0, w - 2))
        fy0 = max(0, min(fy0, h - 2))
        fx1 = max(fx0 + MIN_ROI, min(fx1, w))
        fy1 = max(fy0 + MIN_ROI, min(fy1, h))
        self.roi = (fx0, fy0, fx1 - fx0, fy1 - fy0)
        self._draw_roi()
        self._set_status(f"已更新取景框：x={fx0}, y={fy0}, 宽={fx1-fx0}, 高={fy1-fy0}")

    def reset_roi(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo("提示", "请先打开摄像头或图片。")
            return
        h, w = self.current_frame.shape[:2]
        r = default_roi(w, h)
        self.roi = r.as_tuple()
        self._draw_roi()
        self._set_status("已恢复默认取景框。也可以按住鼠标左键拖动重新框选。")

    # -------------------------------------------------------------- 摄像头
    def start_camera(self) -> None:
        if self.worker is not None:
            self._set_status("摄像头已经在运行。")
            return
        self._camera_error_shown = False
        self._is_demo = False
        self.mode = "camera"
        self.worker = CameraWorker(self.camera_index)
        self.worker.start()
        self._set_status("正在打开摄像头…")
        self.root.after(200, self._tick)

    def stop_camera(self) -> None:
        if self.worker is not None:
            try:
                self.worker.stop()
            except Exception:
                pass
            self.worker = None
        if self.mode == "camera":
            self.mode = "idle"
        self._set_status("摄像头已关闭。")

    def _tick(self) -> None:
        if not self._alive:
            return
        worker = self.worker
        if worker is None:
            return
        if worker.error and not self._camera_error_shown:
            self._camera_error_shown = True
            self._set_status(worker.error.replace("\n", " "))
            try:
                messagebox.showwarning("摄像头不可用", worker.error)
            except Exception:
                pass
            self.stop_camera()
            return
        frame = worker.snapshot()
        if frame is not None:
            if self.current_frame is None or self.current_frame.shape != frame.shape:
                h, w = frame.shape[:2]
                self.roi = default_roi(w, h).as_tuple()
            self.current_frame = frame
            self.image_path = None
            self._show_frame(frame)
        self.root.after(33, self._tick)

    # -------------------------------------------------------------- 打开图片
    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择舌象照片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        try:
            img = resize_max(load_image(path))
            self._load_array(img, source=path)
            self._set_status(f"已载入图片：{path}")
        except ImageError as exc:
            messagebox.showwarning("无法打开图片", str(exc))
        except Exception:
            log.exception("打开图片失败")
            messagebox.showerror("错误", "打开图片时出现异常，请查看 logs/app.log。")

    def _load_array(self, img: np.ndarray, source: str = "", is_demo: bool = False) -> None:
        self.stop_camera()
        self.mode = "image"
        self._is_demo = bool(is_demo)
        self.current_frame = img
        self.image_path = source if os.path.exists(source) else None
        h, w = img.shape[:2]
        self.roi = default_roi(w, h).as_tuple()
        self._show_frame(img)

    # -------------------------------------------------------------- 分析
    def analyze_current(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo(
                "还没有画面",
                "请先点击【打开摄像头】或【打开图片】，再进行分析。",
            )
            return
        frame = self.current_frame.copy()

        issues = assess_capture_quality(frame)
        if issues and not self._is_demo:
            ok = messagebox.askyesno(
                "图像质量提示",
                "\n".join("· " + x for x in issues)
                + "\n\n仍要继续分析吗？（结果可能不准确）",
            )
            if not ok:
                return

        self._set_status("正在分析，请稍候…")
        try:
            self.root.update_idletasks()
        except Exception:
            pass

        try:
            seg = segment_tongue(frame, self.roi)
        except TongueNotFoundError as exc:
            messagebox.showwarning("未识别到舌体", str(exc))
            self._set_status("未识别到舌体，请调整取景框或重新拍摄。")
            return
        except ImageError as exc:
            messagebox.showwarning("图像不可用", str(exc))
            return
        except Exception:
            log.exception("分割失败")
            messagebox.showerror("分析失败", "图像分割时出现异常，请查看 logs/app.log。")
            return

        try:
            feats = extract_features(frame, seg)
            matches = evaluate(feats.to_dict(), self.kb)
            report = build_report(
                feats,
                matches,
                self.kb,
                seg=seg,
                image=frame,
                image_path=self.image_path,
            )
            self.last_report = report
            self.last_paths = report.save()
            self._set_report(report.to_text())
            self._show_frame(annotate(frame, seg))
            self._set_status(
                f"分析完成：{report.matches[0].name if report.matches else '无明确倾向'}"
                f" ｜ 报告已保存到 {self.last_paths.get('txt', '')}"
            )
        except FeatureError as exc:
            messagebox.showwarning("分析失败", str(exc))
        except Exception:
            log.exception("分析失败")
            messagebox.showerror(
                "分析失败",
                "分析过程中出现异常。程序没有崩溃，你可以换一张图片再试。\n"
                "详细信息见 logs/app.log。",
            )

    # -------------------------------------------------------------- 报告操作
    def save_report(self) -> None:
        if self.last_report is None:
            messagebox.showinfo("提示", "还没有生成报告，请先进行分析。")
            return
        try:
            target = filedialog.askdirectory(title="选择报告保存目录")
            if not target:
                return
            paths = self.last_report.save(target)
            self._set_status(f"报告已保存到：{target}")
            messagebox.showinfo("保存成功", f"报告已保存：\n{paths['txt']}")
        except Exception:
            log.exception("保存报告失败")
            messagebox.showerror("保存失败", "保存报告时出现异常，请查看 logs/app.log。")

    def print_report(self) -> None:
        if self.last_report is None:
            messagebox.showinfo("提示", "还没有生成报告，请先进行分析。")
            return
        path = self.last_paths.get("html") or str(
            reports_dir() / f"{self.last_report.rid}.html"
        )
        try:
            if not os.path.exists(path):
                self.last_paths = self.last_report.save()
                path = self.last_paths["html"]
            if sys.platform.startswith("win"):
                try:
                    os.startfile(path, "print")  # type: ignore[attr-defined]
                    self._set_status("已调用系统打印。")
                    return
                except Exception:
                    pass
            webbrowser.open(Path(path).resolve().as_uri())
            self._set_status("已在浏览器中打开可打印报告，按 Ctrl+P 打印。")
        except Exception:
            log.exception("打印失败")
            messagebox.showerror("打印失败", "无法打开报告，请手动到 reports 目录查看。")

    def open_reports_dir(self) -> None:
        try:
            d = str(reports_dir())
            if sys.platform.startswith("win"):
                os.startfile(d)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception:
            messagebox.showinfo("报告目录", str(reports_dir()))


def launch(
    demo: Optional[str] = None, kb_path: Optional[str] = None, camera_index: int = 0
) -> int:
    setup_logging()
    try:
        kb = load_knowledge_base(kb_path)
    except KnowledgeBaseError as exc:
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("知识库错误", str(exc))
            root.destroy()
        except Exception:
            print(exc, file=sys.stderr)
        return 2

    root = tk.Tk()
    try:
        TongueApp(root, kb, demo=demo, camera_index=camera_index)
        root.mainloop()
    except Exception:
        log.exception("界面运行失败")
        try:
            messagebox.showerror(
                "程序异常",
                "界面运行时出现异常，程序已安全退出。\n详情见 logs/app.log。",
            )
        except Exception:
            pass
        return 1
    return 0
