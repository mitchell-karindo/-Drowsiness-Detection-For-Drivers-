# Drowsiness Detection App memakai Streamlit and Roboflow
import streamlit as st
import cv2
import numpy as np
from roboflow import Roboflow
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ultralytics import YOLO

# Set page config
st.set_page_config(page_title="Drowsiness Detection", layout="wide")

# Load YOLO model
model = YOLO('best 50 epoch.pt')

# UI Elements
st.title("Sleep/Drowsiness Recognition System")
st.write("Detect drowsiness in real-time or from uploaded videos/images")

# Sidebar controls
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider("Overlap Threshold", 0.0, 1.0, 0.5, 0.01)

# Function to process image
def process_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image_rgb, conf=confidence_threshold, iou=overlap_threshold)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image, results

# Main functionality
option = st.radio("Select input type:", ("Upload Image", "Upload Video", "Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded Image", use_column_width=True, channels="BGR")
        if st.button("Detect Drowsiness"):
            processed_image, results = process_image(image.copy())
            st.image(processed_image, caption="Processed Image", use_column_width=True, channels="BGR")
            drowsy = 0
            alert = 0
            for result in results:
                for cls in result.boxes.cls:
                    label = model.names[int(cls)]
                    if label.lower() in ['drowsy', 'sleepy']:
                        drowsy += 1
                    elif label.lower() in ['alert', 'awake']:
                        alert += 1
            st.write(f"**Detection Results:** {drowsy} drowsy, {alert} alert")

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = output_file.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if st.button("Process Video"):
            progress_bar = st.progress(0)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame, _ = process_image(frame)
                out.write(processed_frame)
                stframe.image(processed_frame, channels="BGR")
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(min(progress, 1.0))
                time.sleep(0.05)
            cap.release()
            out.release()
            st.success("Video processing completed!")
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(label="Download Processed Video", data=video_bytes, file_name="processed_video.mp4", mime="video/mp4")

elif option == "Webcam":
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.drowsy_detected = False
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed_img, predictions = process_image(img)
            drowsy = 0
            for prediction in predictions:
                for cls in prediction.boxes.cls:
                    label = model.names[int(cls)]
                    if label.lower() in ['drowsy', 'sleepy']:
                        drowsy += 1
            self.drowsy_detected = drowsy > 0
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
    st.header("Live Webcam Detection (Browser only)")
    ctx = webrtc_streamer(
        key="drowsiness",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if ctx.video_processor:
        if ctx.video_processor.drowsy_detected:
            st.warning("⚠️ Mengantuk terdeteksi! Please stay alert!")
        else:
            st.success("Tidak ada tanda mengantuk.")

st.sidebar.markdown("---")
st.sidebar.info("Using YOLO model for drowsiness detection")
