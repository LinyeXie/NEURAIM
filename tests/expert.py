from ultralytics import YOLO

model = YOLO(
    r"E:\Floder\VsCodeFile\PyCodes\yolo_enemy_detector\runs\yolo12_l\yolo12l_.pt",
    task="detect",
)

model.export(
    format="engine",
    device=0,
    imgsz=512,
    quantize=16,   # FP16，替代 half=True
    dynamic=False,
)