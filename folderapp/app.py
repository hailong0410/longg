import streamlit as st
import cv2
import numpy as np
import av
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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
        # Dùng riêng 1 MTCNN để lấy bbox khi vẽ
        self.mtcnn = MTCNN(
            keep_all=False,
            post_process=False,
            min_face_size=40,
            device=device
        )

    def process_frame(self, frame_bgr: np.ndarray):
        # chuyển sang RGB để detect
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # detect face crop
        boxes, probs = self.mtcnn.detect(rgb, landmarks=False)
        if boxes is None or probs is None:
            return frame_bgr, None

        # lọc xác suất >0.9
        faces = []
        filtered_boxes = []
        for box, prob in zip(boxes, probs):
            if prob and prob > 0.9:
                x1, y1, x2, y2 = box.astype(int)
                faces.append(rgb[y1:y2, x1:x2])
                filtered_boxes.append((x1, y1, x2, y2))

        if not faces:
            return frame_bgr, None

        # predict emotions
        emotions, _ = self.recognizer.predict_emotions(faces, logits=False)

        # vẽ bbox + label lên frame gốc
        for (x1, y1, x2, y2), emo in zip(filtered_boxes, emotions):
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr, emo, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        return frame_bgr, emotions[0]

class EmotionTransformer(VideoTransformerBase):
    def __init__(self, model_name, device):
        self.recognizer = RealTimeEmotionRecognizer(
            model_name=model_name,
            device=device
        )

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_out, emotion = self.recognizer.process_frame(img_bgr)
        return av.VideoFrame.from_ndarray(img_out, format="bgr24")

def main():
    st.title("🎥 Nhận diện cảm xúc realtime từ webcam")

    # Sidebar chọn model + device
    device = st.sidebar.selectbox("Chọn thiết bị", ["cpu", "cuda"])
    model_name = st.sidebar.selectbox("Chọn model", get_model_list())

    # Stream video qua WebRTC
    webrtc_streamer(
        key="emotion",
        video_transformer_factory=lambda: EmotionTransformer(model_name, device),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
