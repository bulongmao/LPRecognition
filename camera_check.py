"""摄像头诊断: 尝试多个后端与索引, 报告哪个能用。"""
import cv2

backends = [
    ("AVFOUNDATION", getattr(cv2, "CAP_AVFOUNDATION", 1200)),
    ("DEFAULT/ANY", cv2.CAP_ANY),
]

print("OpenCV:", cv2.__version__)
found = []
for bname, bflag in backends:
    for idx in range(0, 4):
        cap = cv2.VideoCapture(idx, bflag)
        opened = cap.isOpened()
        ok, frame = (False, None)
        if opened:
            ok, frame = cap.read()
        shape = frame.shape if ok and frame is not None else None
        cap.release()
        status = "OK" if ok else ("opened-but-no-frame" if opened else "fail")
        print(f"backend={bname:14s} index={idx}  -> {status}  frame={shape}")
        if ok:
            found.append((bname, bflag, idx))

print()
if found:
    b, f, i = found[0]
    print(f"可用: backend={b} index={i}  -> 在主程序里用 CAM_INDEX={i}, CAM_BACKEND={b}")
else:
    print("没有任何摄像头可用。请检查:")
    print("  1) 系统设置 > 隐私与安全性 > 摄像头 -> 允许 运行它的程序")
    print("     (终端 Terminal / iTerm / PyCharm, 打开开关后需完全退出该程序再重开)")
    print("  2) 是否有其他 App(FaceTime/Zoom/相机) 正占用摄像头, 先关掉。")
    print("  3) 若通过 SSH/远程会话运行, 无法访问本地摄像头。")
