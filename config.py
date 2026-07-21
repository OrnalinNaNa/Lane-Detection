from pathlib import Path
import json
from os import getenv

# โฟลเดอร์ที่เก็บคลิปทดสอบกล้องหน้า/หลัง
CLIPS_DIR = Path("/Users/gkanakorn/Documents/Image/Project/ClipTest")

# รายการ "คู่คลิป" สำหรับโหมด Dev (เล่นคลิปแทนกล้องจริง)
# แต่ละ tuple = (คลิปกล้องหน้า, คลิปกล้องหลัง)
TEST_PAIRS = [
    (CLIPS_DIR / "fcam1.mp4", CLIPS_DIR / "rcam1.mp4"),
    (CLIPS_DIR / "fcam2.mp4", CLIPS_DIR / "rcam2.mp4"),
    (CLIPS_DIR / "fcam3.mp4", CLIPS_DIR / "rcam3.mp4"),
]

# ขนาดเฟรมที่เราจะใช้แสดง/ประมวลผล (กว้าง x สูง)
OUT_W, OUT_H = 1920, 1080

#
# -------------------- โซนของกล้องหลัง: ตรวจรถ --------------------
# กำหนด ROI (สี่เหลี่ยมคางหมู) ที่ด้านล่างของภาพหลัง
# เพื่อบอกว่า "ให้ตรวจรถเฉพาะโซนนี้" จะได้เร็วและลด false positive
# หมายเหตุ: เฉพาะโหมด Realtime เท่านั้นที่ใช้ ROI ขอบบนสูงขึ้น (VEHICLE_ROI_REAR_RT)
# ส่วนโหมด Dev (เล่นคลิป) ให้ใช้ ROI ต่ำกว่า (VEHICLE_ROI_REAR)
VEHICLE_ROI_REAR  = [
    (int(OUT_W*0.25), int(OUT_H*0.72)),  # ขอบบนเดิม (ต่ำกว่า) ใช้สำหรับโหมด Dev (เล่นคลิป)
    (int(OUT_W*0.75), int(OUT_H*0.72)),
    (int(OUT_W*0.75), OUT_H),
    (int(OUT_W*0.25), OUT_H),
]

# -------------------- ROI กล้องหลัง (Realtime): กลางเฟรมจริง ๆ --------------------
# ใช้เฉพาะตอน Realtime เท่านั้น
_rt_roi_w = 0.70   # กว้าง 70% ของเฟรม
_rt_roi_h = 0.35   # สูง 35% ของเฟรม

_center_x = OUT_W * 0.5
_center_y = OUT_H * 0.5

_rt_half_w = int(OUT_W * _rt_roi_w * 0.5)
_rt_half_h = int(OUT_H * _rt_roi_h * 0.5)

VEHICLE_ROI_REAR_RT = [
    (int(_center_x - _rt_half_w), int(_center_y - _rt_half_h)),  # ซ้ายบน
    (int(_center_x + _rt_half_w), int(_center_y - _rt_half_h)),  # ขวาบน
    (int(_center_x + _rt_half_w), int(_center_y + _rt_half_h)),  # ขวาล่าง
    (int(_center_x - _rt_half_w), int(_center_y + _rt_half_h)),  # ซ้ายล่าง
]

# mapping พิกเซลแนวตั้ง (y) -> ระยะทางจริง (เมตร) ของกล้องหลัง
# ตีความแบบง่ายๆ: จุดล่างสุดของภาพ (y=1080) ห่างประมาณ 5 m
# ส่วนสูงขึ้นไป (y=600) ห่างประมาณ 20 m
REAR_Y2M  = ((1080, 5.0), (600, 20.0))

# ระยะที่ถือว่า "อันตราย" ถ้ารถเข้ามาใกล้กว่านี้ให้ขึ้นเตือน/สีแดง/เสียง
DANGER_M = 9.0

# พื้นที่ขั้นต่ำของกล่องรถที่ cascade ตรวจเจอ (เล็กกว่านี้มักเป็น noise)
MIN_VEHICLE_AREA = 2000

# -------------------- โซนของกล้องหน้า: ตรวจเลน --------------------
# จุด 4 จุดของ ROI ด้านหน้า (มุมล่างซ้าย, ล่างขวา, บนขวา, บนซ้าย)
# เอาไปทำ perspective transform → BEV เพื่อดูเลนให้ตรง
LANE_ROI_FRONT = [(200, 1080), (1720, 1080), (1100, 650), (820, 650)]

# ค่าเริ่มต้นของ HSV ที่จะใช้กรองสีเส้นเลน (โหลดจากไฟล์ calib ได้ในภายหลัง)
HSV_LOW  = (0, 0, 180)
HSV_HIGH = (179, 70, 255)

# ถ้าความครอบคลุมของเส้นบน mask ต่ำกว่าค่านี้จะมีแนวโน้มถูกจัดเป็น dashed
DASHED_COVERAGE_THRESH = 0.55

# -------------------- การหาตัว cascade ของรถ --------------------
# โฟลเดอร์ที่เก็บไฟล์ cascade ตรวจรถ (haarcascade_car.xml ฯลฯ)
DEFAULT_CASCADE_DIR = Path("/Users/gkanakorn/Documents/Image/Project/Final_Code/cascades")

# อนุญาตให้ตั้ง path ผ่าน environment variable ด้วย (สะดวกตอนย้ายเครื่อง)
CASCADE_CAR_ENV = getenv("CASCADE_CAR_PATH")

# ชื่อไฟล์ที่เป็นไปได้ของ cascade ตรวจรถ ลองหาเรียงตามลิสต์นี้
CASCADE_CAR_CANDIDATES = [
    "haarcascade_car.xml",
    "cars.xml",
    "haarcascade_cars.xml",
]

# -------------------- โหลดค่าคาลิเบรตเลนจากไฟล์ --------------------
# ถ้ามีไฟล์ lane_calib.json ให้โหลดมาทับค่าด้านบน
# เช่น ปรับ HSV ให้เข้ากับสภาพแสงจริง, ปรับจุด BEV ให้ตรงกับกล้องจริง
CALIB_PATH = Path("/Users/gkanakorn/Documents/Image/Project/Final_Code/lane_calib.json")
if CALIB_PATH.exists():
    try:
        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            _cal = json.load(f)
        # ถ้าในไฟล์มีเก็บ hsv_low / hsv_high → ใช้ค่านั้นแทนของเดิม
        if "hsv_low" in _cal and "hsv_high" in _cal:
            HSV_LOW  = tuple(int(v) for v in _cal["hsv_low"])
            HSV_HIGH = tuple(int(v) for v in _cal["hsv_high"])
        # ถ้ามีจุด BEV (tl, bl, tr, br) → ใช้แทน LANE_ROI_FRONT ได้เลย
        if "bev_pts" in _cal:
            _bp = _cal["bev_pts"]
            BEV_TL = tuple(_bp.get("tl", [222,387]))
            BEV_BL = tuple(_bp.get("bl", [70,472]))
            BEV_TR = tuple(_bp.get("tr", [400,380]))
            BEV_BR = tuple(_bp.get("br", [538,472]))
    except Exception as e:
        # ถ้าโหลดไม่ได้ ให้แจ้งเตือน แต่โปรแกรมยังทำงานต่อด้วยค่าดีฟอลต์
        print(f"[config] Warn: failed to load lane_calib.json: {e}")


# ---------------- Camera (Realtime) Settings ----------------
# ดัชนีกล้อง USB สำหรับโหมด Realtime (กำหนดเองได้ตอนรัน)
# 0 = กล้องตัวแรก, 1 = ตัวที่สอง
CAMERA_FRONT_IDX = 0
CAMERA_REAR_IDX  = 1

# ความละเอียด/เฟรมเรตที่ตั้งค่าให้กล้อง (ขึ้นกับฮาร์ดแวร์ว่าจะรองรับจริงแค่ไหน)
CAMERA_SIZE = (1280, 720)
CAMERA_FPS  = 30

# เปิด/ปิดหน้าต่าง debug ตอนเริ่มโปรแกรม (ระหว่างรันกด 'd' สลับได้)
SHOW_DEBUG_WINDOWS = False

# ประมวลผลจริงทุกกี่เฟรม (เช่น 3 = อ่านกล้องทุกเฟรม แต่ประมวลผลทุกเฟรมที่ 3 เพื่อลดโหลด CPU)
PROCESS_EVERY_N = 3

# ---------------- กลับภาพกล้อง ----------------
FRONT_FLIP_V = True     # กล้องหน้ากลับหัว (กลับบน-ล่าง)
FRONT_FLIP_H = True    # False กล้องหน้าไม่ต้องกลับซ้ายขวา ,True ใช้ตอน realtime
REAR_FLIP_H  = False    # True กล้องหลัง flip ซ้ายขวาแบบกระจก , False ใช้ตอน realtime
REAR_FLIP_V  = False    # False กล้องหลังไม่ต้องกลับบนล่าง , True กลับบนล่าง