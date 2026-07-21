import cv2
from typing import Tuple

# พาเลตสีที่ใช้ทั้งโปรเจกต์ (BGR)
# ถูก import ไปใช้ใน lane_state เพื่อบอกว่า RED = อันตราย, GREEN = ปลอดภัย
COLORS = {
    'GREEN':  (0, 200, 0),    # เขียวปลอดภัย
    'RED':    (0, 0, 255),    # แดงอันตราย
    'ORANGE': (0, 140, 255),  # ส้ม (ยังไม่ได้ใช้เยอะ แต่เผื่อแจ้งเตือนอื่น)
    'CYAN':   (255, 255, 0)   # ฟ้า เห็นชัดบนพื้นมืด
}


def put_text(img,
             text: str,
             org: Tuple[int, int],
             color: Tuple[int, int, int] = (255, 255, 255),
             scale: float = 0.9,
             thickness: int = 2):
    """
    ฟังก์ชันห่อ cv2.putText ให้ใช้ง่ายและได้สไตล์เดียวกันทั้งโปรเจกต์

    พารามิเตอร์:
        img   : ภาพ BGR ที่จะเขียนข้อความลงไป (in-place)
        text  : ข้อความ
        org   : จุดเริ่ม (x, y)
        color : สีตัวอักษร (BGR)
        scale : ขนาดตัวอักษร
        thickness: ความหนา
    """
    cv2.putText(img, text, org,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness,
                cv2.LINE_AA)


def draw_boxes(img, boxes):
    """
    วาดกรอบรถ + ใส่ label ระยะทางบนภาพด้านหลัง
    กล่องที่ส่งเข้ามาเป็นรูปแบบเดียวกับที่ vehicle.detect() คืนค่า:
        [ ((x,y,w,h), dist_m), ... ]
    """
    for (x, y, w, h), dist_m in boxes:
        # วาดกรอบรอบรถ
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # ข้อความเริ่มต้น
        label = "Car"
        # ถ้ามีระยะทาง (เมตร) ก็ใส่เพิ่ม
        if dist_m is not None:
            try:
                label = f"Car {dist_m:.1f}m"
            except Exception:
                # กันกรณี format ไม่ได้
                label = "Car"

        # วาดข้อความเหนือกล่อง (ถ้ากล่องสูงไปก็เลื่อนลงมานิด)
        put_text(img, label, (x, max(y - 8, 20)), (0, 255, 255), 0.8, 2)


def apply_half_overlay(img: cv2.Mat,
                       left_color: Tuple[int, int, int],
                       right_color: Tuple[int, int, int],
                       alpha: float = 0.16) -> None:
    """
    ทับสีโปร่งแสงแยกซ้าย/ขวาบนภาพ (ใช้แสดงเลนซ้าย-ขวาอันตราย/ปลอดภัย)
    - ถ้าฝั่งไหนส่ง None มา → จะไม่ทับสีนั้น
    - หลังสุดจะวาดเส้นแบ่งกลางภาพให้

    พารามิเตอร์:
        img        : ภาพ BGR ที่จะวาดลงไป (แก้ไข in-place)
        left_color : สีฝั่งซ้าย (BGR) หรือ None
        right_color: สีฝั่งขวา (BGR) หรือ None
        alpha      : ความทึบของ overlay
    """
    H, W = img.shape[:2]

    # copy ภาพไว้สำหรับวาด overlay แยกก่อนค่อย blend
    overlay = img.copy()
    drew = False  # เอาไว้เช็กว่ามีฝั่งไหนวาดจริงไหม

    # วาดสี่เหลี่ยมทึบฝั่งซ้ายบน overlay
    if left_color is not None:
        cv2.rectangle(overlay, (0, 0), (W // 2, H), left_color, -1)
        drew = True

    # วาดสี่เหลี่ยมทึบฝั่งขวาบน overlay
    if right_color is not None:
        cv2.rectangle(overlay, (W // 2, 0), (W, H), right_color, -1)
        drew = True

    # ถ้ามีการวาดอย่างน้อย 1 ฝั่ง → blend กับภาพจริง
    if drew:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    # วาดเส้นแบ่งซ้าย-ขวาให้เห็นพื้นที่แต่ละฝั่งชัด ๆ
    cv2.line(img, (W // 2, 0), (W // 2, H), (220, 220, 220), 1, cv2.LINE_AA)


def draw_side_debug(img: cv2.Mat,
                    left_state: str, right_state: str,
                    left_color: Tuple[int, int, int],
                    right_color: Tuple[int, int, int],
                    dangerL: bool, dangerR: bool) -> None:
    """
    วาดแถบดำด้านบน + ข้อความสถานะเลนซ้าย/ขวา
    ใช้ใน lane_state เพื่อให้คนขับดูออกว่าตอนนี้ระบบมองว่า:
        - ซ้าย: dashed (ok)
        - ขวา: solid (danger)

    พารามิเตอร์:
        img         : ภาพที่จะวาด
        left_state  : สตริงสถานะฝั่งซ้าย  ('solid' / 'dashed' / 'none')
        right_state : สตริงสถานะฝั่งขวา
        left_color  : สีข้อความซ้าย (จะส่งเป็นสี overlay มาเลย ก็จะตรงกัน)
        right_color : สีข้อความขวา
        dangerL     : ถ้าซ้ายอันตรายให้เติม (danger)
        dangerR     : ถ้าขวาอันตรายให้เติม (danger)
    """
    h, w = img.shape[:2]
    pad = 10

    # ประกอบข้อความซ้าย/ขวา
    left_txt  = f"Left: {left_state}" + (" (danger)" if dangerL else " (ok)")
    right_txt = f"Right: {right_state}" + (" (danger)" if dangerR else " (ok)")

    bar_h = 32  # ความสูงของแถบดำด้านบน

    # วาดแถบดำโปร่ง ๆ ด้านบนก่อน
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h + 2 * pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # ข้อความฝั่งซ้าย
    put_text(img, left_txt, (pad, pad + 22), left_color, 0.7, 2)

    # ข้อความฝั่งขวา ต้องวัดความกว้างก่อนเพื่อจัดชิดขวา
    (tw, th), _ = cv2.getTextSize(right_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    put_text(img, right_txt, (w - pad - tw, pad + 22), right_color, 0.7, 2)