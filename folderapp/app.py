import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

def recognize_faces(frame: np.ndarray, device: str):
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
    boxes, probs = mtcnn.detect(frame, landmarks=False)
    if boxes is None or probs is None:
        return []
    boxes = boxes[probs > 0.9]
    faces = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        faces.append(frame[y1:y2, x1:x2, :])
    return faces

class RealTimeEmotionRecognizer:
    def __init__(self, model_name=None, device="cpu"):
        if model_name is None:
            model_name = get_model_list()[0]
        self.device = device
        self.recognizer = EmotiEffLibRecognizer(
            engine="onnx",
            model_name=model_name,
            device=device
        )
        self.mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)

    def process_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = recognize_faces(rgb, self.device)
        if len(faces) == 0:
            return frame_bgr, None  # Không phát hiện mặt

        emotions, _ = self.recognizer.predict_emotions(faces, logits=False)
        # Vẽ bounding box và label lên frame
        for bbox, emo in zip(self.mtcnn.detect(rgb)[0], emotions):
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, emo, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
        return frame_bgr, emotions[0]  # Trả về frame đã vẽ và emotion đầu tiên

def main():
    st.title("🎥 Nhận diện cảm xúc realtime từ webcam")

    device = st.sidebar.selectbox("Chọn thiết bị", ["cpu", "cuda"])
    model_name_list = get_model_list()
    model_name = st.sidebar.selectbox("Chọn model", model_name_list)

    recognizer = RealTimeEmotionRecognizer(model_name=model_name, device=device)

    run = st.checkbox("Bật webcam và nhận diện")
    FRAME_WINDOW = st.image([])

    cap = None
    if run:
        cap = cv2.VideoCapture(0)  # Mở webcam mặc định
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Không lấy được hình ảnh từ webcam.")
                break

            frame_processed, emotion = recognizer.process_frame(frame)
            FRAME_WINDOW.image(frame_processed)

            # Dừng vòng lặp nếu user tắt checkbox
            if not run:
                break
    if cap:
        cap.release()

if __name__ == "__main__":
    main()
