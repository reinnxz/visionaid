import os
import time
import json
import queue
import threading
import numpy as np
import cv2
import pyttsx3
from PIL import Image
import torch

from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from vosk import Model as VoskModel, KaldiRecognizer
import sounddevice as sd

# ---------------- CONFIG ----------------
VOSK_MODEL_PATH = os.path.expanduser("~/models/vosk/vosk-model-small-en-us-0.15")
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FOCAL_LENGTH = 700.0
KNOWN_WIDTHS = {"person": 0.5, "car": 1.8, "bicycle": 0.5, "motorbike": 0.6, "truck": 2.5, "bus": 2.5}
DETECTION_CONF = 0.45

COLLISION_DISTANCE = 2.0
AREA_TRIGGER_RATIO = 0.25
OBSTACLE_COOLDOWN = 3.0
VOICE_TRIGGER_COOLDOWN = 1.0

# Voice commands
LISTEN_PHRASES = ["what am i seeing", "describe", "what do i see", "stop", "start"]

# VAD params
AUDIO_RATE = 16000
AUDIO_BLOCKSIZE = 2000
RMS_THRESHOLD = 250
VOICE_BAND_MIN = 300
VOICE_BAND_MAX = 3000
VOICE_BAND_RATIO_THRESHOLD = 0.12

# ---------------- Initialization ----------------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)
tts_engine.setProperty("volume", 1.0)

yolo_model = YOLO("yolov8n.pt")

device = torch.device("cpu")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

if not os.path.isdir(VOSK_MODEL_PATH):
    raise SystemExit(f"VOSK model not found at {VOSK_MODEL_PATH}")
vosk_model = VoskModel(VOSK_MODEL_PATH)

audio_q = queue.Queue()
listening_flag = False
last_voice_trigger = 0.0
last_obstacle_alert = 0.0
caption_busy = threading.Event()
caption_request_event = threading.Event()
tts_lock = threading.Lock()

# control flags
system_active = True     # True = obstacle detection ON
stop_requested = False   # True = STOP mode engaged

# ---------------- TTS ----------------
def speak_text(text, force=False):
    """
    Speaks text using pyttsx3.
    If force=True, speak even when in STOP mode (used for descriptions).
    """
    if stop_requested and not force:
        return  # mute normal detection messages when stopped

    def _speak(t):
        try:
            with tts_lock:
                tts_engine.say(t)
                tts_engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)

    threading.Thread(target=_speak, args=(text,), daemon=True).start()

# ---------------- VAD callback ----------------
def audio_callback(indata, frames, time_info, status):
    try:
        audio_data = np.frombuffer(indata, dtype=np.int16).astype(np.float32)
    except Exception:
        return

    rms = np.sqrt(np.mean(np.square(audio_data)))
    if rms < RMS_THRESHOLD:
        return

    fft = np.fft.rfft(audio_data)
    fft_mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio_data), d=1.0 / AUDIO_RATE)
    total_energy = np.sum(fft_mag) + 1e-9
    voice_band_mask = (freqs >= VOICE_BAND_MIN) & (freqs <= VOICE_BAND_MAX)
    voice_energy = np.sum(fft_mag[voice_band_mask])
    ratio = voice_energy / total_energy

    if ratio >= VOICE_BAND_RATIO_THRESHOLD:
        audio_q.put(bytes(indata))

# ---------------- VOSK listener ----------------
def vosk_listener_loop(trigger_callback):
    global listening_flag, last_voice_trigger, stop_requested, system_active
    try:
        rec = KaldiRecognizer(vosk_model, AUDIO_RATE)
        while True:
            data = audio_q.get()
            if data is None:
                break
            if rec.AcceptWaveform(data):
                try:
                    res = json.loads(rec.Result())
                    text = res.get("text", "").strip().lower()
                except Exception:
                    text = ""
                if text:
                    now = time.time()
                    if now - last_voice_trigger < VOICE_TRIGGER_COOLDOWN:
                        continue
                    last_voice_trigger = now

                    if "stop" in text:
                        stop_requested = True
                        system_active = False
                        speak_text("Detection stopped. Say 'describe' to describe surroundings or 'start' to resume.", force=True)
                        continue

                    if "start" in text:
                        stop_requested = False
                        system_active = True
                        speak_text("Resuming detection and voice alerts.", force=True)
                        continue

                    if "describe" in text or "what am i seeing" in text or "what do i see" in text:
                        caption_request_event.set()
                        continue
    except Exception as e:
        print("VOSK listener error:", e)

# ---------------- BLIP caption ----------------
def run_caption_and_speak(snapshot_bgr):
    if caption_busy.is_set():
        return
    caption_busy.set()
    try:
        pil_img = Image.fromarray(cv2.cvtColor(snapshot_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
        if pil_img.width > 384:
            pil_img = pil_img.resize((384, int(384 * pil_img.height / pil_img.width)))
        inputs = blip_processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_length=64)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        msg = f"You are seeing: {caption}" if caption else "I can't describe that clearly."
        speak_text(msg, force=True)  # force voice output even in STOP mode
    except Exception as e:
        print("Caption error:", e)
    finally:
        caption_busy.clear()

# ---------------- Start audio stream ----------------
def start_audio_stream():
    try:
        stream = sd.RawInputStream(
            samplerate=AUDIO_RATE,
            blocksize=AUDIO_BLOCKSIZE,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        )
        stream.start()
        return stream
    except Exception as e:
        print("Failed to start audio stream:", e)
        return None

# ---------------- Main detection loop ----------------
def main_loop():
    global last_obstacle_alert
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {CAMERA_INDEX}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        print("Vision Aid started. Say 'start', 'stop', or 'describe'. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            proc_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            if system_active:
                results = yolo_model(proc_frame, conf=DETECTION_CONF, stream=True)
                nearest_obj = None
                nearest_dist = float("inf")
                area_trigger = False

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        name = yolo_model.names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1
                        area_ratio = (w * h) / (FRAME_WIDTH * FRAME_HEIGHT)
                        cv2.rectangle(proc_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        if name in KNOWN_WIDTHS and w > 0:
                            dist = (KNOWN_WIDTHS[name] * FOCAL_LENGTH) / float(w)
                        else:
                            dist = None

                        cx = (x1 + x2) // 2
                        dir_text = "ahead"
                        if cx < FRAME_WIDTH / 3:
                            dir_text = "left"
                        elif cx > 2 * FRAME_WIDTH / 3:
                            dir_text = "right"

                        label = name + (f" {dist:.1f}m" if dist else "")
                        cv2.putText(proc_frame, label, (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                        if area_ratio > AREA_TRIGGER_RATIO:
                            area_trigger = True

                        if dist and dist < nearest_dist:
                            nearest_dist = dist
                            nearest_obj = (name, dist, dir_text)

                now = time.time()
                if not stop_requested:
                    collision = False
                    message = None
                    if nearest_obj and nearest_obj[1] <= COLLISION_DISTANCE:
                        collision = True
                        message = f"{nearest_obj[0]} {nearest_obj[1]:.1f} meters {nearest_obj[2]}"
                    elif area_trigger:
                        collision = True
                        message = "Obstacle very close ahead"

                    if collision and (now - last_obstacle_alert > OBSTACLE_COOLDOWN):
                        last_obstacle_alert = now
                        speak_text("Warning: " + message)

            # Overlay mode info
            mode_text = "RUNNING MODE" if system_active else "STOP MODE"
            color = (0, 255, 0) if system_active else (0, 0, 255)
            cv2.putText(proc_frame, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            # Handle description requests
            if caption_request_event.is_set() and not caption_busy.is_set():
                caption_request_event.clear()
                snapshot = frame.copy()
                threading.Thread(target=run_caption_and_speak, args=(snapshot,), daemon=True).start()

            cv2.imshow("Vision Aid", proc_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print("Main loop error:", e)
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    stream = start_audio_stream()
    if stream:
        vosk_thread = threading.Thread(target=vosk_listener_loop,
                                       args=(lambda: caption_request_event.set(),), daemon=True)
        vosk_thread.start()
        main_loop()
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
    print("Exiting program.")
# Vision Aid main script
# Paste your working code here (the full version with STOP/START/DESCRIBE support)
