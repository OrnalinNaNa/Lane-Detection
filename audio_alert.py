import os
import math
import numpy as np
import platform
import subprocess
import tempfile
import threading
import time

# พยายามนำเข้าไลบรารี simpleaudio สำหรับเล่นเสียง (ถ้ามี)
try:
    import simpleaudio as sa
except Exception:
    sa = None

# ถ้าเป็น macOS (Darwin) ให้ปิด simpleaudio เพราะ macOS จะใช้ afplay แทน
if platform.system() == "Darwin":
    sa = None

# ตัวแปรควบคุม: ถ้าเซ็ต environment variable AUDIO_OFF=1 → ปิดเสียงทั้งหมด
_AUDIO_OFF = os.environ.get("AUDIO_OFF", "0") == "1"

# Sampling rate เริ่มต้น (44100 Hz)
_DEF_SR = 44100


# ==========================
# 🔊 ฟังก์ชันสร้างเสียง beep
# ==========================
def _make_beep(freq=880.0, dur=0.20, sr=_DEF_SR):
    """
    สร้างคลื่นเสียง Sine Wave สำหรับ beep ด้วยความถี่ (freq) และระยะเวลา (dur)
    - freq: ความถี่ของเสียง (Hz)
    - dur: ระยะเวลาเสียง (วินาที)
    - sr: sampling rate
    """
    # สร้างแกนเวลา (ตัวอย่างจุดข้อมูลเสียง)
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # สร้างคลื่น sine wave (0.3 = ลดความดังให้อยู่ในระดับปลอดภัย)
    wave = 0.3 * np.sin(2 * math.pi * freq * t)

    # ทำ fade-in และ fade-out เพื่อให้เสียงไม่ “กระแทก”
    fade = np.linspace(0, 1, int(0.02 * sr))
    wave[:fade.size] *= fade             # ช่วงต้น fade-in
    wave[-fade.size:] *= fade[::-1]      # ช่วงท้าย fade-out

    # แปลงข้อมูลเป็น 16-bit PCM (มาตรฐานเสียง)
    pcm = (wave * 32767).astype(np.int16)

    # คืนค่าข้อมูลเสียง (byte data) และ sampling rate
    return pcm.tobytes(), sr


# ==========================
# 🎚️ ฟังก์ชันเล่นเสียง beep แบบกำหนดเอง (ความถี่/เวลา)
# ==========================
def play_beep_custom(freq: float = 880.0, dur: float = 0.3):
    """
    เล่นเสียง beep ที่กำหนดเองได้:
    - freq: ความถี่ (Hz)
    - dur: ระยะเวลา (วินาที)
    ใช้ในโปรเจกต์นี้สำหรับ:
        🔸 ฝั่งซ้าย (lane ซ้ายแดง) → 800 Hz
        🔸 ฝั่งขวา (lane ขวาแดง) → 1000 Hz
    """
    if _AUDIO_OFF:
        return

    # สร้างเสียงใหม่ด้วยพารามิเตอร์ที่ระบุ
    data, sr = _make_beep(freq=freq, dur=dur)

    # เล่นเสียงบน macOS ด้วย afplay (เช่นเดียวกับด้านบน)
    if platform.system() == "Darwin":
        try:
            import wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                name = tf.name
            with wave.open(name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframesraw(data)
            subprocess.Popen(["afplay", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # ลบไฟล์หลังใช้งาน
            def _cleanup(path):
                time.sleep(2.5)
                try:
                    os.remove(path)
                except Exception:
                    pass
            threading.Thread(target=_cleanup, args=(name,), daemon=True).start()
            return
        except Exception:
            pass

    # ถ้าใช้ simpleaudio ได้ (Windows / Linux)
    if sa is not None:
        try:
            sa.play_buffer(data, 1, 2, sr)
            return
        except Exception:
            pass
    return