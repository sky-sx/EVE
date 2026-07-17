"""
EVE 总控制器
- 启动屏幕捕获（30fps，mss）
- YOLO26n 实时目标检测（GPU）
- OpenCV 窗口实时显示（缩放到固定尺寸）
- ESC 退出
"""
import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from input.screen_capture import ScreenCapture
from core.yolo26.detector import YOLODetector

_DISPLAY_MAX_W = 1280
_DISPLAY_MAX_H = 720
_DETECT_EVERY_N = 2  # 每 N 帧做一次检测，其余帧复用上次结果

_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255),
]


def resize_to_fit(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def draw_detections(img: np.ndarray, detections, scale: float = 1.0) -> np.ndarray:
    for d in detections:
        color = _COLORS[d.class_id % len(_COLORS)]
        x1, y1 = int(d.x1 * scale), int(d.y1 * scale)
        x2, y2 = int(d.x2 * scale), int(d.y2 * scale)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = y1 - th - 4 if y1 - th - 4 > 0 else y1 + th + 4
        cv2.rectangle(img, (x1, label_y - th - 2), (x1 + tw + 2, label_y + 2), color, -1)
        cv2.putText(img, label, (x1 + 1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main():
    print("[EVE] ========== EVE 总控制器 ==========")

    # -- 捕获 --
    print("[EVE] 初始化屏幕捕获...")
    cap = ScreenCapture(monitor_index=1, fps=30)

    # -- 检测器 --
    print("[EVE] 加载 YOLO26n (GPU)...")
    detector = YOLODetector(model_path="yolo26n.pt", conf=0.25, iou=0.45)
    if not detector.load():
        return

    # -- 窗口 --
    win_name = "EVE - YOLO26n"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(win_name, _DISPLAY_MAX_W, _DISPLAY_MAX_H)

    # -- 启动 --
    cap.start()
    print("[EVE] 运行中 | ESC 退出")
    print("-" * 50)

    last_stat = time.time()
    frame_idx = 0
    cached_result = None
    cache_scale = 1.0

    try:
        while cap.running:
            frame = cap.get_latest_frame()
            if frame is None:
                cv2.waitKey(5)
                continue

            frame_idx += 1
            orig_h, orig_w = frame.shape[:2]
            bgr = frame[:, :, :3]  # BGRA → BGR

            # --- 显示帧（缩放） ---
            display = resize_to_fit(bgr, _DISPLAY_MAX_W, _DISPLAY_MAX_H)
            disp_h, disp_w = display.shape[:2]
            scale_x = disp_w / orig_w
            scale_y = disp_h / orig_h

            # --- YOLO 检测（跳帧策略） ---
            detect_ms = 0.0
            detect_count = 0
            if detector.loaded and frame_idx % _DETECT_EVERY_N == 0:
                result = detector.detect(frame)
                cached_result = result
                cache_scale = scale_x
                detect_ms = result.inference_time_ms
                detect_count = len(result.detections)
                draw_detections(display, result.detections, scale_x)
            elif cached_result is not None:
                detect_ms = cached_result.inference_time_ms
                detect_count = len(cached_result.detections)
                draw_detections(display, cached_result.detections, cache_scale)

            # --- 信息栏 ---
            bar_h = 32
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (disp_w, bar_h), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)
            info = (f"Capt: {cap.avg_fps:.1f}fps | Infer: {detect_ms:.1f}ms | "
                    f"Objs: {detect_count} | Frame: {frame_idx} | "
                    f"{orig_w}x{orig_h} -> {disp_w}x{disp_h}")
            cv2.putText(display, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow(win_name, display)

            # --- 控制台 ---
            now = time.time()
            if now - last_stat >= 1.0:
                parts = [f"capt={cap.avg_fps:.1f}fps", f"frames={frame_idx}"]
                if detector.loaded:
                    parts.append(f"infer_avg={detector.avg_inference_time_ms:.1f}ms")
                    parts.append(f"objs={detect_count}")
                print(f"[EVE] {' | '.join(parts)}")
                last_stat = now

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[EVE] ESC 退出")
                break

    except KeyboardInterrupt:
        print("\n[EVE] Ctrl+C 中断")
    finally:
        cap.stop()
        if detector:
            detector.unload()
        cv2.destroyAllWindows()
        print("[EVE] 退出完成")


if __name__ == "__main__":
    main()
