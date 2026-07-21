import cv2
import numpy as np

# เก็บพารามิเตอร์เส้นของเฟรมก่อนหน้า (ซ้าย/ขวา)
# เอาไว้ใช้เวลาฟรมใหม่หาเส้นไม่ได้ จะได้ไม่กระพริบ
_prev_left_fit = None
_prev_right_fit = None


def make_coords(img, line_params):
    """แปลงสมการเส้นตรง y = m x + c ให้เป็นพิกัด 2 จุดในภาพ (x1,y1,x2,y2)
    เพื่อให้เอาไปวาดเส้นยาวจากล่างจอขึ้นไปกลางจอได้

    - img: ภาพที่ใช้อ้างอิงความสูง
    - line_params: (slope, intercept) หรือ (m, c)
    """
    slope, intercept = line_params

    # y จุดล่างสุดของภาพ (ขอบล่าง) → ทำให้เส้นลากถึงล่างจอ
    y1 = img.shape[0]

    # y จุดที่สูงขึ้นไปหน่อย (ประมาณ 3/5 ของความสูง) → เส้นจะไม่ยาวเกิน
    y2 = int(y1 * 3 / 5)

    # คำนวณ x กลับจาก y ด้วยสูตร x = (y - c) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    # คืนเป็นอาเรย์ 4 ค่า ใช้กับ cv2.line ได้เลย
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def avg_slope_intercept(img, lines):
    """รวมเส้นหลายเส้นให้เป็นเส้นซ้าย 1 เส้น และขวา 1 เส้นที่นิ่งขึ้น

    หลักการ:
    1. ถ้าไม่มีเส้นใหม่เลย → ใช้ของเก่า (temporal hold)
    2. แยกเส้นฝั่งซ้าย/ขวาจาก sign ของ slope
       - slope < 0 → ฝั่งซ้าย
       - slope > 0 → ฝั่งขวา
    3. เอาเส้นแต่ละฝั่งมาเฉลี่ย (average) เพื่อให้เส้นนิ่ง
    4. แปลง (slope, intercept) กลับเป็นพิกัด 2 จุดด้วย make_coords()

    คืนค่า:
    - array ขนาด (2,4): [[x1,y1,x2,y2], [x1,y1,x2,y2]] สำหรับซ้ายและขวา
    - หรือ None ถ้ายังไม่มีข้อมูลพอ
    """
    global _prev_left_fit, _prev_right_fit

    # ถ้าเฟรมนี้หาเส้นไม่เจอเลย → ลองใช้ค่าของเฟรมก่อนหน้า
    if lines is None:
        if _prev_left_fit is None or _prev_right_fit is None:
            # ยังไม่เคยมีค่าเก่า → บอกว่าไม่มี
            return None
        # มีค่าเก่า → สร้างพิกัดเส้นจากค่าเก่าแล้วคืนเลย
        return np.array([
            make_coords(img, _prev_left_fit),
            make_coords(img, _prev_right_fit)
        ])

    # มีเส้นเข้ามา → แยกซ้าย/ขวา
    left_fit, right_fit = [], []
    for l in lines:
        x1, y1, x2, y2 = l.reshape(4)

        # fit เส้นตรง 1 เส้นจากจุดปลาย 2 จุด → ได้ slope (m) กับ intercept (c)
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

        # ถ้า m < 0 ให้ถือว่าเป็นเส้นฝั่งซ้าย, m > 0 เป็นฝั่งขวา
        (left_fit if slope < 0 else right_fit).append((slope, intercept))

    # ถ้าฝั่งใดฝั่งหนึ่งหายไป (เช่น เห็นแต่เส้นซ้าย) → ลองใช้ค่าจากเฟรมก่อนหน้า
    if not left_fit or not right_fit:
        if _prev_left_fit is None or _prev_right_fit is None:
            return None
        return np.array([
            make_coords(img, _prev_left_fit),
            make_coords(img, _prev_right_fit)
        ])

    # เฉลี่ยเส้นที่ได้ของแต่ละฝั่ง → ได้เส้นซ้าย/ขวาที่นิ่ง
    _prev_left_fit  = np.average(left_fit,  axis=0)
    _prev_right_fit = np.average(right_fit, axis=0)

    # แปลงเป็นพิกัดจุด 2 จุดต่อฝั่งสำหรับวาดจริง
    return np.array([
        make_coords(img, _prev_left_fit),
        make_coords(img, _prev_right_fit)
    ])