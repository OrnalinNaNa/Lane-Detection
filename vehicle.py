import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

class VehicleDetector:
    def __init__(
        self,
        min_area: int = 1500,
        use_cascade: bool = True,
        cascade_path: Optional[str] = None,
        cascade_scale: float = 1.1,
        cascade_neighbors: int = 3,
        cascade_min_size: Tuple[int, int] = (48, 48),
        upscale: float = 1.5,
        dual_pass: bool = True,
    ) -> None:
        # ------------ โหมดสำรอง ถ้าโหลด cascade ไม่ได้ ------------
        # ถ้า True จะไม่ใช้ Haar cascade แต่จะคืนกรอบกลางจอแทน
        self.simple_box = False

        self.min_area = int(min_area)
        self.cascade_scale = float(cascade_scale)
        self.cascade_neighbors = int(cascade_neighbors)
        self.cascade_min_size = (int(cascade_min_size[0]), int(cascade_min_size[1]))
        self.upscale = float(upscale)
        self.dual_pass = bool(dual_pass)
        self.cascade = None
        self.loaded_path = None

        # ลองหาไฟล์ cascade จากหลาย ๆ ที่เหมือนเดิม
        try:
            from config import CASCADE_CAR_ENV, DEFAULT_CASCADE_DIR, CASCADE_CAR_CANDIDATES
        except Exception:
            CASCADE_CAR_ENV = os.environ.get("CASCADE_CAR_PATH")
            DEFAULT_CASCADE_DIR = None
            CASCADE_CAR_CANDIDATES = ["cars.xml", "haarcascade_car.xml", "haarcascade_cars.xml"]

        candidates: List[str] = []
        if cascade_path:
            candidates.append(cascade_path)
        if CASCADE_CAR_ENV:
            candidates.append(CASCADE_CAR_ENV)
        if DEFAULT_CASCADE_DIR is not None:
            for name in CASCADE_CAR_CANDIDATES:
                candidates.append(os.path.join(str(DEFAULT_CASCADE_DIR), name))

        for p in candidates:
            if not p:
                continue
            if os.path.exists(p):
                cc = cv2.CascadeClassifier(p)
                if not cc.empty():
                    self.cascade = cc
                    self.loaded_path = p
                    break

        # ---- ตรงนี้คือจุดที่เราเปลี่ยน ----
        # เดิมมัน raise IOError เลย ตอนนี้ให้ fallback แทน
        if self.cascade is None:
            print("[VehicleDetector] WARN: Haar cascade not found. Falling back to simple center-box detection.")
            self.simple_box = True
        else:
            print(f"[VehicleDetector] Using Haar cascade: {self.loaded_path}")

    @staticmethod
    def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        x1 = max(ax, bx); y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        ua = aw * ah + bw * bh - inter
        return inter / float(ua)

    @staticmethod
    def _crop_roi(frame_bgr: np.ndarray, roi_poly: Optional[List[Tuple[int,int]]]):
        if roi_poly is None:
            return frame_bgr, 0, 0
        rect = cv2.boundingRect(np.array(roi_poly, dtype=np.int32))
        x, y, w, h = rect
        x2, y2 = x + w, y + h
        h0, w0 = frame_bgr.shape[:2]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w0, x2), min(h0, y2)
        if x2 <= x or y2 <= y:
            return frame_bgr, 0, 0
        return frame_bgr[y:y2, x:x2], x, y

    def _passes_shape_filters(self, box: Tuple[int,int,int,int], H_full: int, y_off: int = 0) -> bool:
        x, y, w, h = box
        if w * h < self.min_area:
            return False
        ar = w / float(max(h, 1))
        if not (0.6 <= ar <= 4.5):
            return False
        y_center_abs = (y + h / 2.0) + float(y_off)
        if y_center_abs < (0.35 * H_full):
            return False
        return True

    def _detect_cascade(self, roi_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        # ถ้ามี cascade ก็ตรวจแบบปกติ
        if self.upscale and self.upscale != 1.0:
            img = cv2.resize(roi_bgr, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_LINEAR)
            scale = self.upscale
        else:
            img = roi_bgr
            scale = 1.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        except Exception:
            gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3,3), 0)

        boxes = []
        carsA = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.cascade_scale,
            minNeighbors=self.cascade_neighbors,
            minSize=self.cascade_min_size,
        )
        boxes.extend(list(carsA))

        if self.dual_pass:
            sfB = max(1.03, min(1.08, self.cascade_scale - 0.02))
            nbB = max(2, self.cascade_neighbors - 1)
            msB = (max(32, int(self.cascade_min_size[0] * 0.8)),
                   max(32, int(self.cascade_min_size[1] * 0.8)))
            carsB = self.cascade.detectMultiScale(
                gray,
                scaleFactor=sfB,
                minNeighbors=nbB,
                minSize=msB,
            )
            boxes.extend(list(carsB))

        if scale != 1.0:
            out = []
            for (x, y, w, h) in boxes:
                out.append((int(round(x/scale)), int(round(y/scale)),
                            int(round(w/scale)), int(round(h/scale))))
            boxes = out
        return boxes

    def detect(self, frame_bgr: np.ndarray, y2m, danger_m: float, roi=None):
        H_full, W_full = frame_bgr.shape[:2]

        # --------- โหมด fallback: ไม่มี cascade → วาดกรอบกลางจอ/ROI ---------
        if getattr(self, "simple_box", False):
            if roi is not None:
                rect = cv2.boundingRect(np.array(roi, dtype=np.int32))
                rx, ry, rw, rh = rect
                box_w = int(rw * 0.5)
                box_h = int(rh * 0.55)
                bx = rx + (rw - box_w) // 2
                by = ry + (rh - box_h) // 2
            else:
                box_w = int(W_full * 0.28)
                box_h = int(H_full * 0.35)
                bx = (W_full - box_w) // 2
                by = int(H_full * 0.45)
            dist_m = 8.0
            danger = dist_m < float(danger_m)
            return [((bx, by, box_w, box_h), dist_m)], danger, None
        # ------------------------------------------------------------------

        # 1) crop ตาม ROI ก่อน
        roi_img, x_off, y_off = self._crop_roi(frame_bgr, roi)

        # ถ้ามี ROI เราให้ผ่านง่ายขึ้นหน่อย
        dynamic_min_area = self.min_area
        if roi is not None:
            dynamic_min_area = int(self.min_area * 0.6)

        # 2) detect ใน ROI
        raw_boxes = self._detect_cascade(roi_img)

        # 3) กรองรูปร่าง/ขนาด
        cand = []
        for b in raw_boxes:
            x, y, w, h = b
            if w * h < dynamic_min_area:
                continue
            ar = w / float(max(h, 1))
            if not (0.45 <= ar <= 5.0):
                continue
            if roi is None:
                # ถ้าไม่ได้บังคับกรอบสีฟ้าภายหลัง ก็ใช้ฟิลเตอร์แนวตั้งปกติ
                if not self._passes_shape_filters(b, H_full, y_off=y_off):
                    continue
            cand.append(b)

        # 4) รวมกล่องที่ทับกันเยอะ ๆ
        merged = []
        for b in cand:
            keep = True
            for i, m in enumerate(merged):
                if self._iou(b, m) > 0.4:
                    x = min(m[0], b[0]); y = min(m[1], b[1])
                    x2 = max(m[0] + m[2], b[0] + b[2])
                    y2 = max(m[1] + m[3], b[1] + b[3])
                    merged[i] = (x, y, x2 - x, y2 - y)
                    keep = False
                    break
            if keep:
                merged.append(b)

        # 4.5) ***เพิ่มตรงนี้***: ถ้ามี ROI (กรอบสีฟ้า) ให้คัดเฉพาะกล่องที่อยู่ในกรอบจริง ๆ
        if roi is not None and len(merged) > 0:
            # ทำ mask ของกรอบสีฟ้า
            roi_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [np.array(roi, dtype=np.int32)], 255)

            filtered = []
            for (x, y, w, h) in merged:
                # กล่องตอนนี้ยังเป็นพิกัดใน ROI → บวก offset กลับไปก่อน
                abs_x = x + x_off
                abs_y = y + y_off
                # ตัด mask เฉพาะส่วนกล่อง
                sub = roi_mask[abs_y:abs_y + h, abs_x:abs_x + w]
                if sub.size == 0:
                    continue
                ratio = float(cv2.countNonZero(sub)) / float(sub.size)
                # อยู่ในกรอบสีฟ้า >= 50% ถือว่าโอเค
                if ratio >= 0.5:
                    # เก็บเป็นพิกัดจริงในภาพเต็มไว้เลย
                    filtered.append((abs_x, abs_y, w, h))
            # ใช้รายการที่ผ่านแล้วแทน
            merged = filtered
            # เมื่อเราบวก offset แล้ว ด้านล่างไม่ต้องบวกซ้ำอีก ให้ตั้ง offset เป็น 0
            x_off, y_off = 0, 0

        # 5) แปลงเป็นพิกัดเต็ม + คำนวณระยะ
        def y_to_m(y_px: float, mapping):
            (y1, d1), (y2, d2) = mapping
            if y1 == y2:
                return min(d1, d2)
            y_clamp = max(min(y_px, max(y1, y2)), min(y1, y2))
            t = (y_clamp - y1) / float(y2 - y1)
            return d1 + t * (d2 - d1)

        boxes = []
        closest_d = None

        for (x, y, w, h) in merged:
            abs_x = x + x_off
            abs_y = y + y_off
            cy = abs_y + h
            dist_m = y_to_m(cy, y2m)
            boxes.append(((abs_x, abs_y, w, h), dist_m))
            if closest_d is None or dist_m < closest_d:
                closest_d = dist_m

        danger = (closest_d is not None and closest_d < float(danger_m))
        return boxes, danger, None