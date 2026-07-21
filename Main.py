import cv2
import numpy as np
import config as _cfg
import argparse
import sys

# บอก OpenCV ให้ใช้แค่ 1 thread (กันมันกิน CPU เองเยอะเกิน)
cv2.setNumThreads(1)

from pathlib import Path

# ดึงค่าคงที่ต่าง ๆ จาก config
from config import (
    TEST_PAIRS,      # รายการคู่คลิปที่ใช้ทดสอบโหมด Dev
    OUT_W, OUT_H,    # ขนาดเฟรมมาตรฐานที่เราจะรีไซส์ไปให้เท่ากัน
    MIN_VEHICLE_AREA # พื้นที่ขั้นต่ำของรถที่ตรวจเจอ (ส่งให้ VehicleDetector)
)

# โมดูลช่วยเปิดวิดีโอ/กล้อง และอ่านคู่เฟรม
from sources import open_video, read_pair, open_camera
# ตัวตรวจรถด้านหลัง (Haar cascade)
from vehicle import VehicleDetector
# path ของไฟล์ cascade สำหรับรถ
from config import CASCADE_CAR_ENV, DEFAULT_CASCADE_DIR, CASCADE_CAR_CANDIDATES
# ฟังก์ชันใหญ่ที่ประมวลผล “เฟรมคู่” หน้า+หลัง → คืนรูปที่วาด overlay แล้ว
from lane_state import process_frame_pair


def run_single(front_src: Path, rear_src: Path, use_camera=False):
    """
    รัน 1 แหล่งข้อมูล (จะเป็นกล้องจริงหรือคลิปคู่ก็ได้)
    - front_src, rear_src: path ของคลิปกล้องหน้า/หลัง (ถ้า use_camera=False)
    - use_camera=True → เปิดจากกล้องจริงตาม index ใน config
    วนอ่านทีละเฟรม → ส่งเข้า process_frame_pair → แสดงผล → รอปุ่ม q/n/d
    """

    # บอก config ว่าตอนนี้รันจากกล้องจริงหรือเปล่า
    try:
        import config as _cfg
        _cfg.IS_REALTIME = bool(use_camera)
    except Exception:
        pass
    
    # -----------------------------
    # 1) เตรียมตัวตรวจรถด้านหลัง
    # -----------------------------
    # เลือกไฟล์ cascade ที่มีอยู่จริงตัวแรก
    cascade_hint = (
        CASCADE_CAR_ENV  # ถ้าตั้ง env ไว้ให้ใช้ก่อน
        or next(
            (
                str(DEFAULT_CASCADE_DIR / n)
                for n in CASCADE_CAR_CANDIDATES
                if (DEFAULT_CASCADE_DIR / n).exists()
            ),
            None
        )
    )

    # สร้างตัวตรวจรถ โดยระบุ min_area และ path ของ cascade
    vehR = VehicleDetector(
        min_area=MIN_VEHICLE_AREA,
        use_cascade=True,
        cascade_path=cascade_hint
    )

    # -----------------------------
    # 2) เปิด source หน้า/หลัง
    # -----------------------------
    if use_camera:
        # โหมดกล้องจริง → ใช้ค่าจาก config
        from config import CAMERA_FRONT_IDX, CAMERA_REAR_IDX, CAMERA_SIZE, CAMERA_FPS
        front = open_camera(CAMERA_FRONT_IDX, size=CAMERA_SIZE, fps=CAMERA_FPS, threaded=True)
        rear  = open_camera(CAMERA_REAR_IDX,  size=CAMERA_SIZE, fps=CAMERA_FPS, threaded=True)
    else:
        # โหมดคลิป → เปิดไฟล์วิดีโอ
        front = open_video(front_src)
        rear  = open_video(rear_src)

    window_title = "Front (top) / Rear (bottom)"
    try:
        # สร้างหน้าต่างหลัก
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    except Exception as e:
        print(f"namedWindow error: {e}")

    # ดูจาก config ว่าต้องเปิดหน้าต่าง debug มั้ย
    debug_on = bool(getattr(_cfg, 'SHOW_DEBUG_WINDOWS', False))

    # ฟังก์ชันย่อย: เปิดหน้าต่าง debug 2 อัน (ของหน้าแบบไม่มี BEV และแบบ BEV)
    def _open_debug_windows():
        try:
            cv2.namedWindow("Front lane (no BEV)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Front lane (no BEV)", OUT_W, OUT_H*2)
            dummy2 = np.zeros((OUT_H*2, OUT_W, 3), dtype=np.uint8)
            cv2.imshow("Front lane (no BEV)", dummy2)

            cv2.namedWindow("Front lane (overlay & edges)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Front lane (overlay & edges)", OUT_W, OUT_H*2)
            dummy = np.zeros((OUT_H*2, OUT_W, 3), dtype=np.uint8)
            cv2.imshow("Front lane (overlay & edges)", dummy)

            # ต้องมี waitKey(1) ไม่งั้นบางทีหน้าต่างไม่ขึ้น
            cv2.waitKey(1)
        except Exception as e:
            print(f"open debug windows error: {e}")

    # ฟังก์ชันย่อย: ปิดหน้าต่าง debug
    def _close_debug_windows():
        try:
            cv2.destroyWindow("Front lane (no BEV)")
        except Exception:
            pass
        try:
            cv2.destroyWindow("Front lane (overlay & edges)")
        except Exception:
            pass

    # ถ้า config บอกว่าเปิด debug ตอนเริ่ม ก็เปิดเลย
    if debug_on:
        _open_debug_windows()

    frame_idx = 0           # นับเฟรม (จะได้เลือกว่าจะประมวลผลทุกกี่เฟรม)
    last_stacked = None     # เก็บเฟรมที่ประมวลผลล่าสุดไว้ (เอามาโชว์ซ้ำได้)

    # -----------------------------
    # 3) ลูปหลัก: อ่านคู่เฟรม → ประมวลผล → แสดง → รอคีย์
    # -----------------------------
    while True:
        # read_pair จะคืนภาพหน้า/หลังที่รีไซส์เป็น OUT_W x OUT_H มาให้
        ok, f, r = read_pair(front, rear, (OUT_W, OUT_H))
        if not ok:
            # วิดีโอหมด / กล้องอ่านไม่ได้ → ออกจากลูป
            break

        # เพื่อเซฟโหลด CPU: ประมวลผลจริงทุก PROCESS_EVERY_N เฟรม
        if frame_idx % getattr(_cfg, 'PROCESS_EVERY_N', 2) == 0:
            # process_frame_pair = ฟังก์ชันใหญ่ที่ทำ lane + car + overlay
            stacked = process_frame_pair(f, r, vehR)
            last_stacked = stacked

        # ถ้าเฟรมนี้ไม่ได้ประมวลผล ก็ใช้เฟรมที่ประมวลผลล่าสุดโชว์แทน
        to_show = last_stacked if last_stacked is not None else stacked
        try:
            cv2.imshow(window_title, to_show)
        except Exception as e:
            print(f"imshow crash: {e}")
            status = "quit"
            break

        # รับคีย์จากคีย์บอร์ด (1 ms)
        key = cv2.waitKey(1) & 0xFF
        frame_idx += 1

        # ปุ่มออก
        if key == 27 or key == ord('q'):
            status = "quit"
            break
        # ปุ่มข้ามคลิป (เฉพาะโหมดเล่นคลิป)
        elif key == ord('n'):
            print("Next clip requested.")
            status = "next"
            break
        # ปุ่มเปิด/ปิด debug ระหว่างรัน
        elif key == ord('d'):
            debug_on = not debug_on
            _cfg.SHOW_DEBUG_WINDOWS = debug_on
            if debug_on:
                print("[debug] windows: ON")
                _open_debug_windows()
            else:
                print("[debug] windows: OFF")
                _close_debug_windows()

    # ถ้าไม่ได้ set status ในลูป ให้ถือว่าจบปกติ
    status = locals().get("status", "end")

    # ปิด resource ต่าง ๆ
    front.release()
    rear.release()
    cv2.destroyAllWindows()
    return status


def run_all_pairs_loop(use_camera: bool = False):
    """
    โหมดวนรันหลายคลิปตาม TEST_PAIRS
    - ถ้าเป็นกล้องจริงก็รันแค่รอบเดียว
    - ถ้าเป็นคลิป ก็วนไปเรื่อย ๆ จนครบหรือกดออก
    """
    if use_camera:
        # โหมดกล้องจริง
        try:
            import config as _cfg
            _cfg.IS_REALTIME = True
        except Exception:
            pass
        status = run_single(Path(""), Path(""), use_camera=True)
        return
    else:
        # โหมดคลิป → เซ็ตเป็น False ไปเลย
        try:
            import config as _cfg
            _cfg.IS_REALTIME = False
        except Exception:
            pass

    # โหมดคลิป: ดึงรายการคู่คลิปจาก config
    from config import TEST_PAIRS
    i = 0
    while 0 <= i < len(TEST_PAIRS):
        f, r = TEST_PAIRS[i]
        status = run_single(f, r, use_camera=False)
        if status == "quit":
            break
        else:
            i += 1


# ------------------------------------------------------------
# จุดเริ่มโปรแกรมจริง
# ------------------------------------------------------------
if __name__ == "__main__":
    # สร้าง parser สำหรับอ่าน argument จาก command line
    parser = argparse.ArgumentParser(description="Front/Rear ADAS demo")
    parser.add_argument("--camera", action="store_true", help="ใช้กล้อง USB แทนคลิปวิดีโอ (Realtime)")
    parser.add_argument("--dev",    action="store_true", help="โหมด Dev: ใช้คลิปจาก TEST_PAIRS ใน config.py")
    parser.add_argument("--front",  type=int, default=None, help="index กล้องด้านหน้า (override)")
    parser.add_argument("--rear",   type=int, default=None, help="index กล้องด้านหลัง (override)")
    parser.add_argument("--pair",   type=int, default=None, help="เลือกคู่คลิป (เริ่มต้นที่ 0). ไม่ใส่ = เล่นวนทุกคู่")
    args, unknown = parser.parse_known_args()

    # helper เล็ก ๆ เอาไว้ถามค่า int แบบมี default
    def _ask_int(prompt: str, default: int) -> int:
        try:
            s = input(f"{prompt} [{default}]: ").strip()
            if s == "":
                return int(default)
            return int(s)
        except Exception:
            print("ใส่ค่าไม่ถูกต้อง ใช้ค่าเดิมแทน")
            return int(default)

    # --------------------------------------------------------
    # 1) กรณีผู้ใช้สั่งผ่าน command line เลย
    # --------------------------------------------------------
    if args.camera:
        # ถ้ามีระบุ index กล้องมาด้วย ก็เขียนทับ config ก่อน
        if args.front is not None or args.rear is not None:
            import config as _cfg
            if args.front is not None:
                _cfg.CAMERA_FRONT_IDX = int(args.front)
            if args.rear is not None:
                _cfg.CAMERA_REAR_IDX  = int(args.rear)
        # แล้วค่อยรันโหมดกล้อง
        run_all_pairs_loop(use_camera=True)
        sys.exit(0)

    if args.dev:
        # โหมด dev: เล่นคลิปจาก TEST_PAIRS
        if args.pair is not None:
            # เลือกเล่นคู่เดียว
            from config import TEST_PAIRS
            idx = max(0, min(int(args.pair), len(TEST_PAIRS)-1))
            f, r = TEST_PAIRS[idx]
            run_single(f, r, use_camera=False)
            sys.exit(0)
        # ไม่ระบุ pair → เล่นวนทุกคู่
        run_all_pairs_loop(use_camera=False)
        sys.exit(0)

    # --------------------------------------------------------
    # 2) ไม่ได้ส่ง args มา → เปิดเมนูถามใน console
    # --------------------------------------------------------
    print("\n========================")
    print("   เลือกโหมดการทำงาน   ")
    print("========================")
    print("1) Realtime (USB Camera)")
    print("2) Dev mode (ใช้คลิปจาก config.TEST_PAIRS)")
    print("Q) ออก")

    choice = input("พิมพ์ตัวเลือก: ").strip().lower()
    if choice == "1":
        # ตั้งค่ากล้องจากผู้ใช้ (แต่ถ้ากด Enter เฉย ๆ ใช้ค่าใน config)
        try:
            import config as _cfg
            print("\n-- ตั้งค่ากล้อง USB (กด Enter เพื่อใช้ค่าเดิมจาก config.py) --")
            _cfg.CAMERA_FRONT_IDX = _ask_int("front index", getattr(_cfg, "CAMERA_FRONT_IDX", 0))
            _cfg.CAMERA_REAR_IDX  = _ask_int("rear  index", getattr(_cfg, "CAMERA_REAR_IDX", 1))
        except Exception as e:
            print("[WARN] ไม่สามารถตั้งค่ากล้องผ่าน config ได้:", e)
        run_all_pairs_loop(use_camera=True)

    elif choice == "2":
        # โหมด Dev: เล่นคลิป
        from config import TEST_PAIRS
        try:
            print(f"\nพบคลิปทั้งหมด {len(TEST_PAIRS)} คู่ (index เริ่มที่ 0)")
            s = input("ระบุ pair index ที่ต้องการ (เว้นว่างเพื่อเล่นทุกคู่): ").strip()
            if s == "":
                # เว้นว่าง → เล่นหมด
                run_all_pairs_loop(use_camera=False)
            else:
                # ระบุเลข → เล่นคู่เดียว
                idx = max(0, min(int(s), len(TEST_PAIRS)-1))
                f, r = TEST_PAIRS[idx]
                run_single(f, r, use_camera=False)
        except Exception as e:
            # ถ้าใส่เลขพลาด → fallback เป็นเล่นทุกคู่
            print("[ERR] ค่า pair ไม่ถูกต้อง/อื่น ๆ -> เล่นทุกคู่แทน:", e)
            run_all_pairs_loop(use_camera=False)

    else:
        print("ออกจากโปรแกรม")