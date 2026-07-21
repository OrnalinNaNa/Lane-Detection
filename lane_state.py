import cv2
import numpy as np
from typing import Tuple

from overlay import draw_boxes, put_text, COLORS, apply_half_overlay, draw_side_debug
from lane_utils import avg_slope_intercept
import config as _cfg
from config import OUT_W, OUT_H, LANE_ROI_FRONT, REAR_Y2M, DANGER_M

# ---------------------------------------------------------------------
# โมดูลนี้ทำหน้าที่:
# 1) ประมวลผลภาพจากกล้องหน้า+หลังในเฟรมเดียว
# 2) ตรวจเลน (ผ่าน BEV + mask + Hough)
# 3) ผสานสถานะเลนกับข้อมูลรถด้านหลัง
# 4) คืนภาพที่มี overlay + ข้อความเตือน + กล่องรถ
# ---------------------------------------------------------------------


def bev_persist(prev_state: str, proposed: str, counter: int,
                solid_need: int = 1, dashed_need: int = 5):
    """ทำให้สถานะเลนจาก BEV ไม่แกว่งง่าย:
    - ถ้าสถานะใหม่ตรงกับของเดิม → รีเซ็ตเคาน์เตอร์
    - ถ้าสถานะเปลี่ยน → นับเฟรมให้ครบก่อนค่อยเปลี่ยนจริง
    - เส้นทึบ (solid) จะยืนยันเร็วกว่าเส้นประ (dashed)
    """
    # ถ้าเหมือนเดิมก็ไม่ต้องเปลี่ยน
    if proposed == prev_state:
        return prev_state, 0
    # ไม่เหมือน → เพิ่มตัวนับ
    counter += 1
    # เส้นทึบต้องการเฟรมน้อยกว่าที่จะยืนยัน
    need = solid_need if proposed == 'solid' else dashed_need
    if counter >= need:
        return proposed, 0
    return prev_state, counter


def hysteresis_update(prev_state: str, proposed: str, counter: int,
                      th_solid_to_dashed=8, th_dashed_to_solid=2,
                      th_none_to_state=3, th_state_to_none=6):
    """ฮิสเทอรีซิสสำหรับสถานะเลนซ้าย/ขวา ลดการแกว่งจากเฟรม-ต่อ-เฟรม"""
    # ถ้าเหมือนเดิม → รีเซ็ตตัวนับ
    if proposed == prev_state:
        return prev_state, 0

    # solid → dashed ต้องการเฟรมมากหน่อย
    if prev_state == 'solid' and proposed == 'dashed':
        counter += 1
        return ('dashed', 0) if counter >= th_solid_to_dashed else (prev_state, counter)

    # dashed → solid เร็วกว่าหน่อย
    if prev_state == 'dashed' and proposed == 'solid':
        counter += 1
        return ('solid', 0) if counter >= th_dashed_to_solid else (prev_state, counter)

    # none → เจอเลน (solid/dashed)
    if prev_state == 'none' and proposed in ('solid', 'dashed'):
        counter += 1
        return (proposed, 0) if counter >= th_none_to_state else (prev_state, counter)

    # กลับไป none (ไม่เห็นเลน)
    if proposed == 'none':
        counter += 1
        return ('none', 0) if counter >= th_state_to_none else (prev_state, counter)

    # กรณีอื่น ๆ ใช้เกณฑ์กลาง ๆ
    counter += 1
    return (proposed, 0) if counter >= 3 else (prev_state, counter)


def lane_sides_state_from_mask(mask_bev: np.ndarray, avg_lines_bev, dashed_thresh: float,
                               gap_solid_max=0.28, gap_dashed_min=0.32,
                               band_thick=24, sample_n=140, half_width=5):
    """ตัดสินสถานะเลน (left_state, right_state) ∈ {'dashed','solid','none'} จาก mask + เส้นเฉลี่ย
    เกณฑ์: coverage ในแถบหนา + อัตราช่องว่างยาวสุดตามแนวเส้น
    """
    H, W = mask_bev.shape[:2]
    left_state, right_state = 'none', 'none'
    if avg_lines_bev is None:
        return left_state, right_state

    # ฟังก์ชันย่อย: วัดว่าเส้นนี้มีสีต่อเนื่องแค่ไหน และมีช่วงว่างยาวแค่ไหน
    def _classify_line(line):
        x1, y1, x2, y2 = map(int, line)

        # วาดแถบหนาตามแนวเส้น แล้วดูว่า mask ทับอยู่กี่ %
        band = np.zeros((H, W), dtype=np.uint8)
        cv2.line(band, (x1, y1), (x2, y2), 255, band_thick, cv2.LINE_AA)
        inter = cv2.bitwise_and(band, mask_bev)
        area = int(np.count_nonzero(band))
        cov = float(np.count_nonzero(inter)) / float(max(1, area))

        # สร้างจุดตามแนวเส้น แล้วสุ่มจุดซ้าย-กลาง-ขวา เพื่อดูว่ามีเลนหรือไม่
        xs = np.linspace(x1, x2, num=max(2, sample_n)).astype(np.float32)
        ys = np.linspace(y1, y2, num=max(2, sample_n)).astype(np.float32)
        dx, dy = (x2 - x1), (y2 - y1)
        ln = max(1e-6, float(np.hypot(dx, dy)))
        nx, ny = (-dy / ln, dx / ln)

        hits = []
        for xi, yi in zip(xs, ys):
            xi, yi = float(xi), float(yi)
            xL = int(round(xi - half_width * nx))
            yL = int(round(yi - half_width * ny))
            xR = int(round(xi + half_width * nx))
            yR = int(round(yi + half_width * ny))
            val = 0
            # ถ้าจุดไหนใน 3 จุดติด mask ถือว่ามีเส้น
            for sx, sy in ((xL, yL), (int(round(xi)), int(round(yi))), (xR, yR)):
                if 0 <= sx < W and 0 <= sy < H and mask_bev[int(sy), int(sx)] > 0:
                    val = 1
                    break
            hits.append(val)

        # หาช่วงที่ว่างยาวที่สุด (ใช้แยก dashed)
        longest = 0
        cur = 0
        for v in hits:
            if v == 0:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        longest_gap_ratio = float(longest) / float(len(hits))

        # ตัดสิน solid/dashed จาก gap และ coverage
        if longest_gap_ratio < gap_solid_max:
            state = 'solid'
        elif (cov < dashed_thresh) and (longest_gap_ratio >= gap_dashed_min):
            state = 'dashed'
        else:
            # ค่าอื่น ๆ ให้เป็น solid ไว้ก่อน
            state = 'solid'

        # ดูว่าเส้นนี้อยู่ซีกซ้ายหรือขวาของภาพ BEV
        side = 'left' if ((x1 + x2) * 0.5) < (W * 0.5) else 'right'
        return side, state, cov, longest_gap_ratio

    # เลือกเส้นที่ดูดีที่สุดของแต่ละฝั่ง (ซ้าย/ขวา)
    best = {'left': (-1.0, 'none', 1.0), 'right': (-1.0, 'none', 1.0)}
    for l in avg_lines_bev:
        side, state, cov, lgr = _classify_line(l)
        score = cov - 0.5 * lgr
        if score > (best[side][0] - 0.5 * best[side][2]):
            best[side] = (cov, state, lgr)
    left_state = best['left'][1]
    right_state = best['right'][1]
    return left_state, right_state


def _rear_side_danger(boxes_with_dist, img_width: int, danger_m: float):
    """ดูว่ารถด้านหลังที่อยู่ในกล่องอยู่ฝั่งซ้ายหรือขวา และเข้าใกล้กว่า danger หรือยัง"""
    mx = img_width * 0.5
    left = False
    right = False
    for (x, y, w, h), dist_m in boxes_with_dist:
        if dist_m is None:
            continue
        if dist_m < float(danger_m):
            cx = x + w * 0.5
            if cx < mx:
                left = True
            else:
                right = True
    return left, right


def process_frame_pair(f_bgr, r_bgr, vehR):
    """ประมวลผลเฟรมคู่ (หน้า+หลัง) แล้วคืนภาพที่รวม overlay เรียบร้อย"""
    global _cfg

    # ---------- จัดทิศทางกล้องตามโหมด ----------
    # is_rt = True  → Realtime (กล้อง USB)
    # is_rt = False → Dev mode (เล่นคลิป)
    is_rt = bool(getattr(_cfg, "IS_REALTIME", False))

    if not is_rt:
        # Dev mode
        # กล้องหน้า: ไม่ flip เลย
        # กล้องหลัง: flip ซ้ายขวาอย่างเดียว ให้ภาพเหมือนกระจกหลัง
        r_bgr = cv2.flip(r_bgr, 1)
    else:
        # Realtime mode
        # กล้องหน้า: flip ซ้ายขวา + flip บนล่าง (เพราะติดตั้งกลับหัวและหันเข้าหาผู้ขับ)
        f_bgr = cv2.flip(f_bgr, 1)  # ซ้าย ↔ ขวา
        f_bgr = cv2.flip(f_bgr, 0)  # บน ↕ ล่าง
        # กล้องหลัง: flip ซ้ายขวาอย่างเดียว ให้เหมือนมองกระจกหลังจริง
        r_bgr = cv2.flip(r_bgr, 1)

    # ค่าเริ่มต้นสถานะเลน
    left_lane_state_bev, right_lane_state_bev = 'none', 'none'
    left_lane_state_ori, right_lane_state_ori = 'none', 'none'

    # ตรวจรถด้านหลัง + เช็กว่าเข้าระยะอันตรายไหม
    rear_roi = _cfg.VEHICLE_ROI_REAR_RT if is_rt and hasattr(_cfg, "VEHICLE_ROI_REAR_RT") else _cfg.VEHICLE_ROI_REAR
    boxesR, dangerR, dbgR = vehR.detect(r_bgr, y2m=REAR_Y2M, danger_m=DANGER_M, roi=rear_roi)

    # copy ไว้วาดของเราเอง
    f_vis = f_bgr.copy()
    r_vis = r_bgr.copy()

    # =========================================================
    # 1) ส่วนของกล้องหน้า: แปลงเป็น BEV แล้วหาสถานะเลน
    # =========================================================
    try:
        # จัดลำดับจุด polygon ให้เป็น tl, bl, tr, br
        def _order_pts(poly):
            pts = np.array(poly, dtype=np.float32)
            idx = np.argsort(pts[:, 1])
            top2, bot2 = pts[idx[:2]], pts[idx[2:]]
            tl = top2[np.argmin(top2[:, 0])]
            tr = top2[np.argmax(top2[:, 0])]
            bl = bot2[np.argmin(bot2[:, 0])]
            br = bot2[np.argmax(bot2[:, 0])]
            return tl, bl, tr, br

        # ใช้จุดจากไฟล์ calib ถ้ามี ไม่งั้นใช้จาก config ปกติ
        if all(hasattr(_cfg, k) for k in ("BEV_TL", "BEV_BL", "BEV_TR", "BEV_BR")):
            tl, bl, tr, br = _cfg.BEV_TL, _cfg.BEV_BL, _cfg.BEV_TR, _cfg.BEV_BR
        else:
            tl, bl, tr, br = _order_pts(LANE_ROI_FRONT)

        # ทำ perspective transform → BEV
        bev_w, bev_h = 640, 480
        M = cv2.getPerspectiveTransform(np.float32([tl, bl, tr, br]),
                                        np.float32([[0, 0], [0, bev_h], [bev_w, 0], [bev_w, bev_h]]))
        bev = cv2.warpPerspective(f_vis, M, (bev_w, bev_h))

        # แปลงสีเพื่อทำ mask
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bev, cv2.COLOR_BGR2LAB)
        L, _, _ = cv2.split(lab)

        # รวม mask สีเส้นถนน (ขาว/เหลือง) + mask จากไฟล์ calib
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200], np.uint8), np.array([179, 80, 255], np.uint8))
        yellow_mask = cv2.inRange(hsv, np.array([15, 60, 120], np.uint8), np.array([40, 255, 255], np.uint8))
        calib_mask = cv2.inRange(hsv, np.array(_cfg.HSV_LOW, np.uint8), np.array(_cfg.HSV_HIGH, np.uint8))
        color_mask = cv2.bitwise_or(white_mask, cv2.bitwise_or(yellow_mask, calib_mask))

        # ตรวจเส้นขอบจาก L channel
        gradx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
        absx = np.absolute(gradx)
        grad_norm = (absx * (255.0 / max(1.0, absx.max()))).astype(np.uint8)
        _, grad_mask = cv2.threshold(grad_norm, 30, 255, cv2.THRESH_BINARY)

        # รวม mask สี + เส้นขอบ แล้วทำให้เนียน
        mask = cv2.bitwise_or(color_mask, grad_mask)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        edges = cv2.Canny(mask, 50, 150)

        # หาเส้นใน BEV
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=120)
        good = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l.reshape(4)
                slope = 999.0 if x2 == x1 else (y2 - y1) / float(x2 - x1)
                if abs(slope) > 0.5:
                    good.append([x1, y1, x2, y2])
        lines = np.array(good).reshape((-1, 1, 4)) if len(good) > 0 else None

        # ทำให้เส้นนิ่งขึ้น
        avg_lines = avg_slope_intercept(bev, lines) if lines is not None else None
        bev_overlay = bev.copy()
        if avg_lines is not None:
            for x1, y1, x2, y2 in avg_lines:
                cv2.line(bev_overlay, (x1, y1), (x2, y2), (255, 0, 255), 10)

        # ตัดสินว่าเลนซ้าย/ขวาเป็นชนิดไหน
        left_lane_state, right_lane_state = lane_sides_state_from_mask(
            mask, avg_lines,
            dashed_thresh=getattr(_cfg, 'DASHED_COVERAGE_THRESH', 0.55)
        )

        # ทำ persistence ของ BEV (เก็บในตัวแปรบน function object)
        if not hasattr(process_frame_pair, "_BEV_STATE_L"):
            process_frame_pair._BEV_STATE_L = 'none'
            process_frame_pair._BEV_STATE_R = 'none'
            process_frame_pair._BEV_CNT_L = 0
            process_frame_pair._BEV_CNT_R = 0

        process_frame_pair._BEV_STATE_L, process_frame_pair._BEV_CNT_L = bev_persist(
            process_frame_pair._BEV_STATE_L, left_lane_state, process_frame_pair._BEV_CNT_L,
            solid_need=getattr(_cfg, 'BEV_SOLID_PERSIST', 1),
            dashed_need=getattr(_cfg, 'BEV_DASH_PERSIST', 5))
        process_frame_pair._BEV_STATE_R, process_frame_pair._BEV_CNT_R = bev_persist(
            process_frame_pair._BEV_STATE_R, right_lane_state, process_frame_pair._BEV_CNT_R,
            solid_need=getattr(_cfg, 'BEV_SOLID_PERSIST', 1),
            dashed_need=getattr(_cfg, 'BEV_DASH_PERSIST', 5))

        left_lane_state_bev, right_lane_state_bev = process_frame_pair._BEV_STATE_L, process_frame_pair._BEV_STATE_R

        # สำหรับ debug
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.putText(bev_overlay, 'BEV raw', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(edges_bgr, 'BEV edges', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # แสดงหน้าต่างดีบักถ้าเปิดไว้
        if getattr(_cfg, 'SHOW_DEBUG_WINDOWS', False):
            h, w = bev_overlay.shape[:2]
            scale = min(OUT_W / float(w), OUT_H / float(h))
            new_w, new_h = int(w * scale), int(h * scale)
            resized_bev = cv2.resize(bev_overlay, (new_w, new_h))
            resized_edges = cv2.resize(edges_bgr, (new_w, new_h))

            canvas_top = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
            canvas_bottom = np.zeros_like(canvas_top)
            x_off = (OUT_W - new_w) // 2
            y_off = (OUT_H - new_h) // 2
            canvas_top[y_off:y_off + new_h, x_off:x_off + new_w] = resized_bev
            canvas_bottom[y_off:y_off + new_h, x_off:x_off + new_w] = resized_edges

            lane_stack = np.vstack([canvas_top, canvas_bottom])
            cv2.imshow("Front lane (overlay & edges)", lane_stack)

    except Exception as e:
        print(f"[lane popup] {e}")
        left_lane_state, right_lane_state = 'none', 'none'

    # =========================================================
    # 2) โหมด debug เพิ่มเติมแบบไม่ใช้ BEV (ดูภาพจริงตรง ๆ)
    # =========================================================
    try:
        if getattr(_cfg, 'SHOW_DEBUG_WINDOWS', False):
            hsv_o = cv2.cvtColor(f_vis, cv2.COLOR_BGR2HSV)
            lab_o = cv2.cvtColor(f_vis, cv2.COLOR_BGR2LAB)
            L_o, _, _ = cv2.split(lab_o)

            white_o = cv2.inRange(hsv_o, np.array([0, 0, 200], np.uint8), np.array([179, 80, 255], np.uint8))
            yellow_o = cv2.inRange(hsv_o, np.array([15, 60, 120], np.uint8), np.array([40, 255, 255], np.uint8))
            calib_o = cv2.inRange(hsv_o, np.array(_cfg.HSV_LOW, np.uint8), np.array(_cfg.HSV_HIGH, np.uint8))
            color_o = cv2.bitwise_or(white_o, cv2.bitwise_or(yellow_o, calib_o))

            gx_o = cv2.Sobel(L_o, cv2.CV_64F, 1, 0, ksize=3)
            gx_o = (np.absolute(gx_o) * (255.0 / max(1.0, np.max(np.absolute(gx_o))))).astype(np.uint8)
            _, gmask_o = cv2.threshold(gx_o, 30, 255, cv2.THRESH_BINARY)

            mask_o = cv2.bitwise_or(color_o, gmask_o)
            mask_o = cv2.GaussianBlur(mask_o, (5, 5), 0)
            kernel_o = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_o = cv2.morphologyEx(mask_o, cv2.MORPH_CLOSE, kernel_o, iterations=3)

            roi_poly = np.array(LANE_ROI_FRONT, dtype=np.int32)
            roi_mask = np.zeros_like(mask_o)
            cv2.fillPoly(roi_mask, [roi_poly], 255)
            mask_o = cv2.bitwise_and(mask_o, roi_mask)

            edges_o = cv2.Canny(mask_o, 50, 150)
            lines_o = cv2.HoughLinesP(edges_o, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=120)

            good_o = []
            if lines_o is not None:
                for l in lines_o:
                    x1, y1, x2, y2 = l.reshape(4)
                    slope = 999.0 if x2 == x1 else (y2 - y1) / float(x2 - x1)
                    if abs(slope) > 0.5:
                        good_o.append([x1, y1, x2, y2])
            lines_o = np.array(good_o).reshape((-1, 1, 4)) if len(good_o) > 0 else None
            avg_lines_o = avg_slope_intercept(f_vis, lines_o) if lines_o is not None else None

            if avg_lines_o is not None:
                _thresh = getattr(_cfg, 'DASHED_COVERAGE_THRESH', 0.55)
                _ls, _rs = lane_sides_state_from_mask(mask_o, avg_lines_o, dashed_thresh=_thresh)
                left_lane_state_ori, right_lane_state_ori = _ls, _rs

            # ฟังก์ชันย่อย: ปรับแสงให้สว่างขึ้น
            def _apply_clahe_bgr_local(img):
                try:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    L, A, B = cv2.split(lab)
                    L2 = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(L)
                    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)
                except Exception:
                    return img

            # ฟังก์ชันย่อย: sharpen นิดหน่อยให้เห็นเส้นชัด
            def _sharpen_local(img, amount=0.30, radius=0.9):
                blur = cv2.GaussianBlur(img, (0, 0), radius)
                return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0.0)

            front_disp = _apply_clahe_bgr_local(f_vis)
            front_overlay = front_disp.copy()
            if avg_lines_o is not None:
                for x1, y1, x2, y2 in avg_lines_o:
                    cv2.line(front_overlay, (x1, y1), (x2, y2), (255, 0, 255), 10)
            front_overlay = _sharpen_local(front_overlay, amount=0.30, radius=0.9)

            edges_bgr_o = cv2.cvtColor(edges_o, cv2.COLOR_GRAY2BGR)
            cv2.putText(front_overlay, 'Front raw (no BEV)', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(edges_bgr_o, 'Front edges (no BEV)', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            stack_no_bev = np.vstack([front_overlay, edges_bgr_o])
            cv2.imshow("Front lane (no BEV)", stack_no_bev)
    except Exception as e:
        print(f"[no-bev popup] {e}")
        left_lane_state, right_lane_state = 'none', 'none'

    # ---------------------------------------------------------
    # รวมสถานะจาก BEV และจากภาพจริง
    # ---------------------------------------------------------
    def _combine_lane_states(bev_state: str, ori_state: str) -> str:
        # ถ้า BEV บอก dashed → เชื่อ BEV
        if bev_state == 'dashed':
            return 'dashed'
        # ถ้าทั้งคู่บอก solid → solid
        if ori_state == 'solid' and bev_state == 'solid':
            return 'solid'
        # ถ้าภาพจริงบอก solid แต่ BEV บอก dashed → dashed
        if ori_state == 'solid' and bev_state == 'dashed':
            return 'dashed'
        # ถ้าภาพจริงบอก solid → เอา solid
        if ori_state == 'solid':
            return 'solid'
        # ถ้า BEV บอก solid → เอา solid
        if bev_state == 'solid':
            return 'solid'
        # ถ้าภาพจริงบอก dashed → เอา dashed
        if ori_state == 'dashed':
            return 'dashed'
        # สุดท้ายไม่มีเลน
        return 'none'

    left_lane_state_final = _combine_lane_states(left_lane_state_bev, left_lane_state_ori)
    right_lane_state_final = _combine_lane_states(right_lane_state_bev, right_lane_state_ori)

    # ใช้ hysteresis ทำให้สถานะเลนสุดท้ายไม่กระพริบ
    if not hasattr(process_frame_pair, "_STABLE_L"):
        process_frame_pair._STABLE_L = 'none'
        process_frame_pair._STABLE_R = 'none'
        process_frame_pair._CNT_L = 0
        process_frame_pair._CNT_R = 0

    process_frame_pair._STABLE_L, process_frame_pair._CNT_L = hysteresis_update(
        process_frame_pair._STABLE_L, left_lane_state_final, process_frame_pair._CNT_L)
    process_frame_pair._STABLE_R, process_frame_pair._CNT_R = hysteresis_update(
        process_frame_pair._STABLE_R, right_lane_state_final, process_frame_pair._CNT_R)

    left_lane_state_stable = process_frame_pair._STABLE_L
    right_lane_state_stable = process_frame_pair._STABLE_R

    # ---------------------------------------------------------
    # วาด ROI กล้องหลังให้เห็นว่าเราตรวจรถเฉพาะโซนล่าง
    # ---------------------------------------------------------
    try:
        import numpy as _np
        rear_roi_for_draw = (_cfg.VEHICLE_ROI_REAR_RT
                             if is_rt and hasattr(_cfg, "VEHICLE_ROI_REAR_RT")
                             else _cfg.VEHICLE_ROI_REAR)
        _rect = cv2.boundingRect(_np.array(rear_roi_for_draw, dtype=_np.int32))
        _x, _y, _w, _h = _rect
        _x = max(0, min(_x, r_vis.shape[1] - 1))
        _y = max(0, min(_y, r_vis.shape[0] - 1))
        _w = max(1, min(_w, r_vis.shape[1] - _x))
        _h = max(1, min(_h, r_vis.shape[0] - _y))
        _overlay = r_vis.copy()
        cv2.rectangle(_overlay, (_x, _y), (_x + _w - 1, _y + _h - 1), (255, 255, 0), -1)
        cv2.addWeighted(_overlay, 0.12, r_vis, 0.88, 0, r_vis)
        cv2.rectangle(r_vis, (_x, _y), (_x + _w - 1, _y + _h - 1), (255, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(r_vis, 'REAR DETECTION ZONE', (_x + 8, max(_y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    except Exception:
        pass

    # ---------------------------------------------------------
    # เช็กว่ารถด้านหลังอยู่ซ้ายหรือขวา และเข้า danger หรือยัง
    # ---------------------------------------------------------
    _, rW = r_bgr.shape[:2]
    dangerL, dangerRside = _rear_side_danger(boxesR, rW, DANGER_M)

    # เลือกสี overlay ตามสถานะเลน + รถด้านหลัง
    def _side_color(lane_state: str, danger_side: bool):
        # อันตรายหรือเป็นเส้นทึบ → แดง
        if danger_side or lane_state == 'solid':
            return COLORS['RED']
        # ตรวจจับได้และเป็นเส้นประ → เขียว
        if lane_state == 'dashed':
            return COLORS['GREEN']
        # ไม่เจอเลน → ไม่ทับสี
        return None

    left_color_overlay = _side_color(left_lane_state_stable, dangerL)
    right_color_overlay = _side_color(right_lane_state_stable, dangerRside)

    # เล่นเสียงเตือนเมื่อเข้าสถานะอันตราย (เปลี่ยนจากไม่แดง → แดง)
    try:
        from audio_alert import play_beep_custom
        left_red = dangerL or (left_lane_state_stable == 'solid')
        right_red = dangerRside or (right_lane_state_stable == 'solid')
        if not hasattr(process_frame_pair, "_prev_left_red"):
            process_frame_pair._prev_left_red = False
            process_frame_pair._prev_right_red = False
        if left_red and not process_frame_pair._prev_left_red:
            play_beep_custom(freq=800.0, dur=0.3)
        if right_red and not process_frame_pair._prev_right_red:
            play_beep_custom(freq=1000.0, dur=0.3)
        process_frame_pair._prev_left_red = left_red
        process_frame_pair._prev_right_red = right_red
    except Exception as e:
        print("[warn beep] error:", e)
        pass

    # ---------------------------------------------------------
    # ทำ smoothing สี overlay ไม่ให้กระพริบเร็วเกิน
    # ---------------------------------------------------------
    if not hasattr(process_frame_pair, '_SMOOTH_PREV_COLOR'):
        process_frame_pair._SMOOTH_PREV_COLOR = (0, 0, 0), (0, 0, 0)
        process_frame_pair._SMOOTH_COUNT = 0
        process_frame_pair._SMOOTH_N = getattr(_cfg, 'SMOOTH_N', 5) if hasattr(_cfg, 'SMOOTH_N') else 5

    commit = True
    if (left_color_overlay, right_color_overlay) != getattr(process_frame_pair, '_SMOOTH_PREV_COLOR',
                                                            ((None, None, None), (None, None, None))):
        process_frame_pair._SMOOTH_COUNT += 1
        commit = (process_frame_pair._SMOOTH_COUNT >= process_frame_pair._SMOOTH_N)

    if commit:
        process_frame_pair._SMOOTH_PREV_COLOR = (left_color_overlay, right_color_overlay)
        process_frame_pair._SMOOTH_COUNT = 0
    else:
        left_color_overlay, right_color_overlay = getattr(process_frame_pair, '_SMOOTH_PREV_COLOR',
                                                          (left_color_overlay, right_color_overlay))

    # ---------------------------------------------------------
    # วาดข้อความ + overlay ลงบนภาพหน้า/หลัง
    # ---------------------------------------------------------
    left_interrupt = (left_color_overlay == COLORS['RED'])
    right_interrupt = (right_color_overlay == COLORS['RED'])
    _GRAY = (200, 200, 200)
    left_text_color = left_color_overlay if left_color_overlay is not None else _GRAY
    right_text_color = right_color_overlay if right_color_overlay is not None else _GRAY

    # แถบบน: แสดงสถานะเลนซ้าย/ขวา
    draw_side_debug(f_vis, left_lane_state_stable, right_lane_state_stable,
                    left_text_color, right_text_color, left_interrupt, right_interrupt)
    draw_side_debug(r_vis, left_lane_state_stable, right_lane_state_stable,
                    left_text_color, right_text_color, left_interrupt, right_interrupt)

    # ทับสีครึ่งจอ
    apply_half_overlay(f_vis, left_color_overlay, right_color_overlay, alpha=0.32)
    apply_half_overlay(r_vis, left_color_overlay, right_color_overlay, alpha=0.32)

    # วาดเฉพาะกล่องรถที่เข้าใกล้
    boxesR_close = [b for b in boxesR if (b[1] is not None and b[1] < DANGER_M)]
    draw_boxes(r_vis, boxesR_close)

    # ข้อความเตือนกลางภาพหลัง
    if dangerR:
        put_text(r_vis, f"Rear Car < {int(DANGER_M)}m", (40, 75), (0, 255, 255), 1.0, 2)

    # สุดท้ายต่อภาพหน้า/หลังเป็นเฟรมเดียว
    stacked = np.vstack([f_vis, r_vis])
    return stacked