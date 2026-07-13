"""
商品防损识别 POC (MacBook 摄像头版)

功能:
  - 实时视频显示
  - 在画面中定义一个"托盘目标区域"(ROI)
  - 物品进入 ROI 时检测并框出 (基于背景建模的前景占位判断)
  - MediaPipe 手部检测: 画出 21 点骨架 + 外框 + 左右手标注
  - 检测到二维码即视为"扫码成功"(模拟扫码)
  - 物品从 ROI 移出但从未扫码成功 -> 声音报警

状态机 (针对整个 ROI, POC 假设一次一个物品):
  IDLE     : ROI 内无物品
  PRESENT  : ROI 内有物品, 尚未扫码
  SCANNED  : ROI 内有物品且已扫码成功
  移出时: 若为 SCANNED -> 正常; 若从未扫码 -> ALARM

操作:
  首帧用鼠标框选托盘区域, 回车/空格确认 (直接回车则用默认居中区域)
  运行中按 q 退出, 按 r 重新框选区域
"""

import time
import subprocess
import threading
import sys
import os

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False


# ---------------- 参数 (可现场标定) ----------------
CAM_INDEX = 0
CAM_BACKEND = None           # None=自动尝试; 或设为 cv2.CAP_AVFOUNDATION
HAND_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
ENABLE_HANDS = True          # 是否启用 MediaPipe 手部检测
FG_AREA_RATIO_ENTER = 0.08   # ROI 内前景占比 > 此值 判定为有物品
FG_AREA_RATIO_EXIT = 0.03    # 前景占比 < 此值 判定物品已移出 (滞回, 防抖)
ENTER_FRAMES = 6             # 连续满足进入条件的帧数
EXIT_FRAMES = 8             # 连续满足移出条件的帧数
ALARM_COOLDOWN = 3.0        # 报警冷却秒数
ALARM_VOLUME = "2.8"       # afplay 音量放大 (越大越爆音/刺耳)
ALARM_SPEAK = True          # 是否叠加语音播报
ALARM_SPEECH = "警告，商品未扫码"
_ALARM_WAV_PATH = None      # 运行时生成的警笛文件


def _make_siren_wav():
    """合成常见的间断蜂鸣警报声(嘀-嘀-嘀)到临时 WAV 文件, 返回路径。"""
    import wave
    import tempfile
    import os
    sr = 44100
    freq = 1000.0        # 蜂鸣音高
    beep = 0.18          # 每声时长
    gap = 0.12           # 间隔
    n_beeps = 4
    seg = []
    tb = np.linspace(0, beep, int(sr * beep), endpoint=False)
    # 正弦音 + 淡入淡出包络, 干净的"嘀"声
    env = np.minimum(1.0, np.minimum(tb, beep - tb) * 40)
    one = np.sin(2 * np.pi * freq * tb) * env * 0.8
    silence = np.zeros(int(sr * gap))
    for _ in range(n_beeps):
        seg.append(one)
        seg.append(silence)
    audio = np.clip(np.concatenate(seg), -1, 1)
    pcm = (audio * 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm]).ravel()
    fd, path = tempfile.mkstemp(suffix="_alarm.wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(stereo.tobytes())
    return path


def _ensure_alarm_wav():
    global _ALARM_WAV_PATH
    if _ALARM_WAV_PATH is None:
        try:
            _ALARM_WAV_PATH = _make_siren_wav()
        except Exception:
            _ALARM_WAV_PATH = "/System/Library/Sounds/Sosumi.aiff"
    return _ALARM_WAV_PATH


def play_alarm():
    """异步播放刺耳警笛(放大音量) + 语音播报, 不阻塞主循环。"""
    sound = _ensure_alarm_wav()

    def _run():
        played = False
        try:
            subprocess.run(
                ["osascript", "-e", "set volume output volume 38"],
                check=False)
        except Exception:
            pass
        # 人声与警笛同时播放(配合): 语音在独立进程并发
        say_proc = None
        try:
            if ALARM_SPEAK:
                say_proc = subprocess.Popen(["say", ALARM_SPEECH])
        except Exception:
            say_proc = None
        try:
            for _ in range(3):
                subprocess.run(
                    ["afplay", "-v", ALARM_VOLUME, sound], check=False)
                # 语音说完就再喊一遍, 保持与警笛同步配合
                if ALARM_SPEAK and (say_proc is None or say_proc.poll() is not None):
                    try:
                        say_proc = subprocess.Popen(["say", ALARM_SPEECH])
                    except Exception:
                        pass
            played = True
        except Exception:
            pass
        if not played:
            sys.stdout.write("\a")
            sys.stdout.flush()
    threading.Thread(target=_run, daemon=True).start()


# 21 个手部关键点之间的连线 (MediaPipe 标准拓扑)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (9, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),# 小指
    (0, 17),                                # 掌根
]


class HandDetector:
    """MediaPipe Tasks 手部检测封装 (VIDEO 模式)。"""

    def __init__(self, model_path, num_hands=2):
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=num_hands,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    def detect(self, bgr_frame, timestamp_ms):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(mp_img, int(timestamp_ms))

    @staticmethod
    def draw(frame, result):
        """在画面上绘制手部骨架、外框与左右手标注; 返回手部外框列表。"""
        boxes = []
        if not result or not result.hand_landmarks:
            return boxes
        H, W = frame.shape[:2]
        for i, lms in enumerate(result.hand_landmarks):
            pts = [(int(l.x * W), int(l.y * H)) for l in lms]
            # 连线
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)
            # 关键点
            for p in pts:
                cv2.circle(frame, p, 3, (255, 0, 255), -1)
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            pad = 12
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(W - 1, x2 + pad), min(H - 1, y2 + pad)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            label = "Hand"
            try:
                if result.handedness and result.handedness[i]:
                    label = result.handedness[i][0].category_name  # Left / Right
            except Exception:
                pass
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes


def select_roi(cap):
    """让用户框选 ROI, 返回 (x, y, w, h)。"""
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("无法读取摄像头画面")
    h, w = frame.shape[:2]
    r = cv2.selectROI("选择托盘目标区域 (拖拽框选, 回车确认)", frame,
                       showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("选择托盘目标区域 (拖拽框选, 回车确认)")
    x, y, ww, hh = r
    if ww == 0 or hh == 0:
        # 默认居中区域
        ww, hh = int(w * 0.4), int(h * 0.4)
        x, y = (w - ww) // 2, (h - hh) // 2
    return int(x), int(y), int(ww), int(hh)


def open_camera():
    """在 macOS 上稳健地打开摄像头: 优先 AVFoundation, 再扫描多个索引。"""
    av = getattr(cv2, "CAP_AVFOUNDATION", 1200)
    if CAM_BACKEND is not None:
        trials = [(CAM_BACKEND, CAM_INDEX)]
    else:
        trials = [(av, CAM_INDEX), (av, 0), (av, 1),
                  (cv2.CAP_ANY, CAM_INDEX), (cv2.CAP_ANY, 0), (cv2.CAP_ANY, 1)]
    for backend, idx in trials:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"摄像头已打开: backend={backend} index={idx}")
                return cap
        cap.release()
    return None


def main():
    cap = open_camera()
    if cap is None or not cap.isOpened():
        print("错误: 打不开摄像头。请依次检查:")
        print("  1) 系统设置 > 隐私与安全性 > 摄像头 -> 允许运行它的程序")
        print("     (Terminal / iTerm / PyCharm); 打开后需完全退出该程序再重开。")
        print("  2) 关闭正在占用摄像头的 App(FaceTime/Zoom/相机 等)。")
        print("  3) 先运行 python camera_check.py 定位可用的后端和索引。")
        print("  4) 不要通过 SSH/远程会话运行, 那样无法访问本地摄像头。")
        return

    roi = select_roi(cap)

    # 背景建模器 (对 ROI 做前景检测)
    bg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=40, detectShadows=True)

    qr = cv2.QRCodeDetector()

    # 手部检测器 (MediaPipe)
    hand_detector = None
    if ENABLE_HANDS and _MP_AVAILABLE and os.path.exists(HAND_MODEL_PATH):
        try:
            hand_detector = HandDetector(HAND_MODEL_PATH, num_hands=2)
            print("MediaPipe 手部检测: 已启用")
        except Exception as e:
            print(f"手部检测初始化失败, 已跳过: {e}")
    elif ENABLE_HANDS and not _MP_AVAILABLE:
        print("未安装 mediapipe, 跳过手部检测。")
    elif ENABLE_HANDS and not os.path.exists(HAND_MODEL_PATH):
        print(f"缺少手部模型 {HAND_MODEL_PATH}, 跳过手部检测。")

    state = "IDLE"
    enter_count = 0
    exit_count = 0
    scanned = False
    last_qr_text = ""
    last_alarm_time = 0.0
    alarm_flash_until = 0.0
    frame_idx = 0

    print("运行中: q 退出 | r 重选区域")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # 镜像, 更符合直觉
        frame_idx += 1
        x, y, w, h = roi
        # 保证 ROI 在画面内
        H, W = frame.shape[:2]
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
        roi = (x, y, w, h)

        roi_img = frame[y:y + h, x:x + w]

        # --- 前景占位检测 ---
        fgmask = bg.apply(roi_img)
        # 去掉阴影(值127)与噪声
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(
            fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fgmask = cv2.morphologyEx(
            fgmask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        fg_ratio = float(np.count_nonzero(fgmask)) / (w * h)

        object_bbox = None
        if fg_ratio > FG_AREA_RATIO_EXIT:
            cnts, _ = cv2.findContours(
                fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c)
                object_bbox = (x + bx, y + by, bw, bh)

        # --- 二维码检测(模拟扫码): 只在 ROI 内检测 ---
        qr_found = False
        try:
            data, pts, _ = qr.detectAndDecode(roi_img)
            if data:
                qr_found = True
                last_qr_text = data
        except cv2.error:
            pass

        # --- 状态机 ---
        present = fg_ratio > FG_AREA_RATIO_ENTER
        absent = fg_ratio < FG_AREA_RATIO_EXIT

        if state == "IDLE":
            enter_count = enter_count + 1 if present else 0
            if enter_count >= ENTER_FRAMES:
                state = "PRESENT"
                scanned = False
                exit_count = 0

        elif state in ("PRESENT", "SCANNED"):
            if qr_found:
                scanned = True
                state = "SCANNED"
            exit_count = exit_count + 1 if absent else 0
            if exit_count >= EXIT_FRAMES:
                # 物品移出
                if not scanned:
                    now = time.time()
                    if now - last_alarm_time > ALARM_COOLDOWN:
                        play_alarm()
                        last_alarm_time = now
                        alarm_flash_until = now + 1.5
                        print(f"[ALARM] 物品未扫码即移出目标区! t={now:.1f}")
                # 重置
                state = "IDLE"
                enter_count = 0
                exit_count = 0
                scanned = False

        # ---------------- 可视化 ----------------
        # ROI 框
        roi_color = (0, 255, 0) if state == "SCANNED" else (
            (0, 165, 255) if state == "PRESENT" else (200, 200, 200))
        cv2.rectangle(frame, (x, y), (x + w, y + h), roi_color, 2)
        cv2.putText(frame, "TARGET ZONE", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

        # 物品框
        if object_bbox is not None:
            ox, oy, ow, oh = object_bbox
            label = "SCANNED" if scanned else "ITEM (unscanned)"
            oc = (0, 255, 0) if scanned else (0, 0, 255)
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), oc, 2)
            cv2.putText(frame, label, (ox, oy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, oc, 2)

        # --- MediaPipe 手部检测 ---
        num_hands = 0
        if hand_detector is not None:
            try:
                ts_ms = int(frame_idx * (1000.0 / 30.0))
                hand_res = hand_detector.detect(frame, ts_ms)
                hand_boxes = HandDetector.draw(frame, hand_res)
                num_hands = len(hand_boxes)
            except Exception:
                num_hands = 0

        # 状态栏
        bar = f"STATE:{state}  fg:{fg_ratio:.2f}  scanned:{scanned}  hands:{num_hands}"
        cv2.putText(frame, bar, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if last_qr_text:
            cv2.putText(frame, f"QR: {last_qr_text[:30]}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # 报警闪烁
        if time.time() < alarm_flash_until:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 12)
            cv2.putText(frame, "!! ALARM: UNSCANNED REMOVAL !!",
                        (W // 2 - 260, H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        cv2.imshow("Loss Prevention POC", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            roi = select_roi(cap)
            bg = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=40, detectShadows=True)
            state = "IDLE"
            enter_count = exit_count = 0
            scanned = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
