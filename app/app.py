
import cv2
import os
import tempfile
import streamlit as st
from roboflow import Roboflow
from datetime import timedelta
import time
import requests
import json

st.set_page_config(layout="wide")
st.title("🚨 Fall Detection Monitor")

# === Sidebar UI for Camera and Webhook ===
st.sidebar.header("📷 Camera & Webhook Settings")

# Load Settings if available
settings = {}
settings_path = "settings.json"
if os.path.exists(settings_path):
    with open(settings_path, "r") as f:
        settings = json.load(f)

def get_setting(key, default=""):
    return settings.get(key, default)

camera_url = st.sidebar.text_input("RTSP / Stream URL", value=get_setting("camera_url"), placeholder="rtsp://... or http://camera-ip/...", key="cam_url")
stored_api_key = st.sidebar.text_input("Roboflow API Key", value=get_setting("api_key", os.getenv("ROBOFLOW_API_KEY")), type="password")
stored_webhook_url = st.sidebar.text_input("Webhook URL (Optional)", value=get_setting("webhook_url"), placeholder="https://your-webhook-endpoint.com", key="webhook_url")

# Save Settings Button
if st.sidebar.button("💾 Save Settings"):
    new_settings = {
        "camera_url": camera_url,
        "api_key": stored_api_key,
        "webhook_url": stored_webhook_url
    }
    with open(settings_path, "w") as f:
        json.dump(new_settings, f)
    st.sidebar.success("✅ Settings saved successfully.")

VIDEO_PATH = "data/pool_fall_video.mp4"

# === Initialize Roboflow ===
rf = Roboflow(api_key=stored_api_key)
project = rf.workspace("fall-ql73x").project("falling-dii3i")
st.success("✅ Roboflow project loaded.")

model = project.version(7).model
if model is None:
    st.error("❌ Model not loaded. Check version or training status.")
    st.stop()
else:
    st.success("✅ Model version loaded.")

# === Load Video or Camera Feed ===
use_camera = camera_url.strip() != ""

if use_camera:
    st.info(f"📡 Connecting to real-time camera stream...")
    cap = cv2.VideoCapture(camera_url)
else:
    if not os.path.exists(VIDEO_PATH):
        st.error("❌ Video not found at 'data/pool_fall_video.mp4'")
        st.stop()
    cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not use_camera else 0

if not use_camera:
    st.info(f"🎬 Video loaded: {total_frames} frames at {fps:.2f} FPS")
    timestamp_slider = st.slider("🕹️ Jump to Second", 0, int(total_frames / fps) - 1, 0)
    play_button = st.button("▶️ Continue Playing From Here")
else:
    timestamp_slider = 0
    play_button = st.button("▶️ Start Live Stream")

frame_display = st.empty()
log_display = st.empty()

falls_detected = []
log_text = ""
frame_count = int(timestamp_slider * fps)

if play_button:
    while True:
        if not use_camera:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Stream ended or cannot read frame.")
            break

        timestamp = frame_count / fps
        readable_time = str(timedelta(seconds=int(timestamp)))
        resized = cv2.resize(frame, (640, 640))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_image_path = tmp_file.name
            cv2.imwrite(temp_image_path, resized)

        try:
            prediction = model.predict(temp_image_path).json()
        except Exception as e:
            os.remove(temp_image_path)
            st.error(f"Prediction error: {e}")
            break

        os.remove(temp_image_path)

        if prediction.get("predictions"):
            label = prediction["predictions"][0]["top"]
            if "fall" in label.lower():
                log_text += f"\n[Frame {frame_count}] 🚨 Fall detected at {readable_time} - Label: '{label}'"
                falls_detected.append(readable_time)

                if stored_webhook_url:
                    try:
                        requests.post(stored_webhook_url, json={"notification": f"Fall detected at {readable_time}", "label": label})
                    except Exception as e:
                        st.warning(f"⚠️ Webhook failed: {e}")
            else:
                log_text += f"\n[Frame {frame_count}] ✅ No fall - classified as '{label}'"
        else:
            log_text += f"\n[Frame {frame_count}] ❓ No prediction returned"

        frame_bgr = cv2.resize(frame, (640, 360))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_display.image(frame_rgb, caption=f"Second {int(timestamp)} | Frame {frame_count}", channels="RGB")

        log_lines = log_text.strip().split("\n")[-10:]
        log_combined = "\n".join(log_lines)
        log_display.code(log_combined, language="bash")

        frame_count += int(fps)
        if not use_camera:
            time.sleep(0.1)
        else:
            time.sleep(1)

cap.release()

st.subheader("🧾 Summary")
if falls_detected:
    for i, t in enumerate(falls_detected, 1):
        st.write(f"{i}. Fall at {t}")
else:
    st.write("✅ No falls detected.")
