from flask import Flask, render_template, Response
import cv2
import numpy as np
import json
import time
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Tải mô hình và nhãn lớp
model = load_model('Mohinh_nhandien.h5')
with open('nhan.json', 'r') as f:
    labels = json.load(f)
labels = {int(k): v for k, v in labels.items()}

# Sử dụng bộ phát hiện khuôn mặt của OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kích thước ảnh đầu vào của mô hình
IMG_SIZE = (64, 64)

# Hàm dự đoán nhãn cho khuôn mặt
def predict_face(face_img):
    # Resize ảnh khuôn mặt về kích thước IMG_SIZE và chuẩn hóa
    face_img = cv2.resize(face_img, IMG_SIZE)
    face_img = face_img / 255.0  # Chuẩn hóa
    face_img = np.expand_dims(face_img, axis=0)  # Thêm chiều batch

    # Dự đoán bằng mô hình
    predictions = model.predict(face_img)
    predicted_class = np.argmax(predictions)  # Lấy nhãn có xác suất cao nhất
    confidence = np.max(predictions)  # Lấy độ tin cậy cao nhất
    return labels[predicted_class], confidence

# Hàm tạo frame video
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w] 

            label, confidence = predict_face(face_img)
            
            if confidence < 0.7:
                label = "unknow"

            # Vẽ bounding box và label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Chuyển đổi frame thành định dạng JPEG để truyền qua HTTP
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    # Trả về giao diện HTML chính
    return render_template('index1.html')

# Endpoint để stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)