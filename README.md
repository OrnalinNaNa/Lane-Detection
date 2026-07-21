# Lane Detection with Front/Rear ADAS Demo

## 📌 Project Overview
โปรเจกต์นี้เป็นตัวอย่างระบบช่วยขับขี่แบบง่าย ๆ ที่รับภาพจากกล้องหน้าและกล้องหลัง แล้วประมวลผลร่วมกันเพื่อ:

- ตรวจจับเส้นเลนบนภาพกล้องหน้า
- ประเมินสถานะเลนว่าเป็น solid / dashed / none
- ตรวจจับรถที่อยู่ด้านหลังด้วย Haar cascade
- แสดงคำเตือนเมื่อรถเข้าใกล้เกิน threshold
- วาด overlay บนภาพเพื่อให้เห็นสถานะเลนและความเสี่ยงได้ชัดเจน

## ✨ Features
- แสดงภาพคู่ Front/Rear แบบสตรีม
- แปลงภาพกล้องหน้าเป็น Bird’s Eye View (BEV)
- ใช้ masking + edge detection + Hough line เพื่อแยกเส้นเลน
- ใช้ Haar cascade ตรวจจับรถด้านหลัง
- รองรับทั้งโหมดเล่นวิดีโอและโหมดกล้องจริง
- มีโหมด debug สำหรับดูภาพขั้นตอนการประมวลผล

## 🛠 Tech Stack
- Python 3.8+
- OpenCV
- NumPy

## 👥 Team Members
- OrnalinNaNa
- peeraphat29
- GKanakorn
- mmommypoko
- nuengruthaiboonmak

## 🚀 How to Run

### 1) การติดตั้ง
ติดตั้ง dependency ที่จำเป็นด้วยคำสั่ง:

```bash
pip install opencv-python numpy
```

### 2) โหมดเล่นคลิปทดสอบ (Dev mode)
ก่อนรัน ให้แก้ค่าใน config.py ให้ชี้ไปที่ไฟล์วิดีโอจริงของคุณ โดยเฉพาะ:
- CLIPS_DIR
- TEST_PAIRS
จากนั้นรัน:

```bash
python3 Main.py --dev
```

### 3) โหมดใช้กล้องจริง

```bash
python3 Main.py --camera
```

ถ้าต้องการระบุ index ของกล้องหน้าและหลัง:

```bash
python3 Main.py --camera --front 0 --rear 1
```

### 4) คีย์ที่ใช้ควบคุมระหว่างรัน
- q หรือ Esc: ออกจากโปรแกรม
- n: ข้ามไปคู่คลิปต่อไป (เฉพาะโหมด Dev)
- d: เปิด/ปิดหน้าต่าง debug

## 📁 Project Structure
- Main.py: จุดเริ่มโปรแกรมและโหมดรันต่าง ๆ
- config.py: ค่า configuration หลัก เช่น path ของคลิป, ค่า ROI, กล้อง, HSV threshold
- lane_state.py: กระบวนการประมวลผลภาพหน้า+หลังและรวม overlay
- lane_utils.py: ฟังก์ชันช่วยสำหรับการหาสมการเส้นเลนและรวมเส้น
- overlay.py: ฟังก์ชันวาดข้อความ/กรอบ/overlay
- sources.py: การเปิดวิดีโอและกล้อง
- vehicle.py: ตัวตรวจจับรถด้านหลัง
- cascades/: ไฟล์ Haar cascade สำหรับตรวจรถ

## ⚠️ Notes
โปรเจกต์นี้กำหนด path แบบ absolute ใน config.py โดยตรง ดังนั้นบนเครื่องอื่น ๆ อาจต้องแก้ค่าต่อไปนี้ให้ตรงกับเครื่องคุณ:
- CLIPS_DIR
- DEFAULT_CASCADE_DIR
- CALIB_PATH
หากไม่พบไฟล์ cascade ที่ตั้งไว้ ระบบจะ fallback เป็นโหมดตรวจจับแบบกรอบกลางจอแทน เพื่อให้โปรแกรมยังรันได้

## 🔧 Configuration Tips
คุณสามารถปรับค่าต่าง ๆ ใน config.py ได้ เช่น:

- HSV_LOW / HSV_HIGH: สำหรับกรองสีเส้นเลน
- LANE_ROI_FRONT: ROI ของภาพหน้าเพื่อสร้าง BEV
- VEHICLE_ROI_REAR / VEHICLE_ROI_REAR_RT: ROI สำหรับตรวจรถ
- DANGER_M: ระยะทางขั้นต่ำที่ถือว่าอันตราย
- PROCESS_EVERY_N: ลดความถี่การประมวลผลเพื่อประหยัด CPU