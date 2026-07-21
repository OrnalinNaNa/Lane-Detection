import cv2
from pathlib import Path
from typing import Tuple
import threading
import time

def open_video(path: Path) -> cv2.VideoCapture:
    """
    เปิดไฟล์วิดีโอจาก path ที่ให้มา แล้วคืนเป็นออบเจ็กต์ cv2.VideoCapture
    ถ้าเปิดไม่ได้ให้ raise ขึ้นไปเลย เพื่อให้โค้ดหลักรู้ว่าคลิปหาย/พาธผิด
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


# --- ThreadedCapture class for low-latency camera capture ---
class ThreadedCapture:
    """
    ตัวห่อกล้องให้อ่านเฟรมใน background thread

    แนวคิด:
        - thread นี้จะอ่านกล้องเรื่อย ๆ แล้วเก็บไว้แค่ "เฟรมล่าสุด"
        - ฝั่ง main/UI ที่เรียก .read() จะได้เฟรมล่าสุดเสมอ (ไม่ต้องรอ decode เฟรมเก่า)
        - ทำให้ realtime ลื่นขึ้น โดยเฉพาะตอนประมวลผลหนัก ๆ

    ใช้กับ open_camera(..., threaded=True)
    """
    def __init__(self, device_index: int, size=(1280, 720), fps: int = 30, backend=None):
        # ใช้ AVFoundation บน macOS จะเสถียรกว่า
        if backend is None:
            try:
                self.cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
            except Exception:
                # ถ้าใช้ backend นี้ไม่ได้ ให้ลองเปิดแบบปกติ
                self.cap = cv2.VideoCapture(device_index)
        else:
            self.cap = cv2.VideoCapture(device_index, backend)

        # ตั้งค่ากล้องเบื้องต้น: ความกว้าง/สูง + fps
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # บังคับใช้ MJPG เพื่อลดภาระการถอดรหัส (บางกล้องจะได้ latency น้อยลง)
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        # ขอให้ driver เก็บ buffer น้อย ๆ (ถ้ากล้องรองรับ) จะได้ไม่ดองเฟรมเก่า
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open camera index {device_index}")

        # ตัวแปรสำหรับแชร์เฟรมระหว่าง thread
        self._lock = threading.Lock()
        self._frame = None      # เฟรมล่าสุด
        self._ok = False        # สถานะอ่านสำเร็จ/ไม่สำเร็จ
        self._stopped = False   # flag หยุด thread

        # สร้างและสตาร์ท thread สำหรับอ่านกล้อง
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        """ฟังก์ชันที่รันอยู่ใน background thread: อ่านกล้องวนไปเรื่อย ๆ"""
        while not self._stopped:
            ok, f = self.cap.read()
            # เก็บเฉพาะเฟรมล่าสุดภายใต้ lock
            with self._lock:
                self._ok = ok
                if ok:
                    self._frame = f
            # ถ้าอ่านไม่ได้ให้พักนิดเดียว ไม่ให้ while วิ่งเปล่า
            if not ok:
                time.sleep(0.005)

    def read(self):
        """
        คืนค่า (ok, frame ล่าสุดหรือ None)
        ให้หน้าตาเหมือน cv2.VideoCapture.read() เพื่อให้ใช้แทนกันได้
        """
        with self._lock:
            if self._frame is None:
                # ยังไม่มีเฟรมแรก
                return self._ok, None
            # copy ออกไป เพื่อไม่ให้คนข้างนอกไปแก้ของใน buffer โดยตรง
            return self._ok, self._frame.copy()

    def release(self):
        """
        หยุด thread และปล่อยกล้อง
        เรียกตอนปิดโปรแกรมหรือเปลี่ยน source
        """
        self._stopped = True
        try:
            self._t.join(timeout=0.2)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


def read_pair(front: cv2.VideoCapture,
              rear: cv2.VideoCapture,
              out_size: Tuple[int, int]):
    """
    อ่านเฟรม 'คู่' จากกล้อง/วิดีโอ 2 ตัว (front, rear)
    แล้วรีไซส์ให้ได้ขนาดเดียวกัน (out_size) พร้อมเติมขอบให้เต็มจอ

    คืนค่า: (ok, front_frame, rear_frame)
    - ok=False ถ้าตัวใดตัวหนึ่งอ่านไม่ได้
    """
    # รองรับทั้ง VideoCapture.read() ปกติ และ ThreadedCapture.read()
    okF, f = front.read()
    okR, r = rear.read()
    if not okF or f is None or not okR or r is None:
        return False, None, None

    target_w, target_h = out_size

    def resize_keep_aspect(img, tw, th):
        """
        รีไซส์ภาพให้พอดีกรอบ (tw, th) โดยคงสัดส่วนเดิมไว้
        ถ้าด้านใดด้านหนึ่งไม่เต็มให้เติมขอบดำ
        """
        ih, iw = img.shape[:2]
        # หาสัดส่วนย่อ/ขยายที่ทำให้ภาพไม่ล้นทั้งสองด้าน
        scale = min(tw / float(iw), th / float(ih))
        new_w, new_h = int(round(iw * scale)), int(round(ih * scale))

        # รีไซส์ภาพ
        resized = cv2.resize(
            img, (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        )

        # คำนวณขอบบนล่างซ้ายขวาให้ภาพอยู่กึ่งกลาง
        top = (th - new_h) // 2
        bottom = th - new_h - top
        left = (tw - new_w) // 2
        right = tw - new_w - left

        # เติมขอบดำให้ภาพมีขนาดเป๊ะเท่าที่ต้องการ
        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return canvas

    # รีไซส์ภาพหน้า/หลังให้เท่ากัน
    f = resize_keep_aspect(f, target_w, target_h)
    r = resize_keep_aspect(r, target_w, target_h)
    return True, f, r


def open_camera(device_index: int,
                size=(1920, 1080),
                fps: int = 30,
                threaded: bool = True):
    """
    เปิดกล้องด้วย index ที่กำหนด
    - ถ้า threaded=True (ค่าเริ่มต้น) → ใช้ ThreadedCapture เพื่อลด latency
    - ถ้า threaded=False → คืน VideoCapture ปกติ
    """
    if threaded:
        # โหมด low-latency: อ่านกล้องใน background thread
        return ThreadedCapture(device_index, size=size, fps=fps)

    # โหมดเดิม (non-threaded) เผื่อใช้กับกล้องแปลก ๆ ที่ thread แล้วพัง
    try:
        cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
    except Exception:
        cap = cv2.VideoCapture(device_index)

    # ตั้งค่ากล้อง
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    # พยายามตั้ง FOURCC เป็น MJPG
    try:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    except Exception:
        pass

    # ลด buffer เท่าที่ทำได้
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap