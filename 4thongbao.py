import cv2
import numpy as np
import json
import datetime
import time
import os
from tensorflow.keras.models import load_model

# Kích thước ảnh đầu vào của mô hình (giống như trong quá trình huấn luyện)
IMG_SIZE = (64, 64)

# Tải mô hình đã huấn luyện và nhãn lớp
model = load_model('Mohinh_nhandien.h5')
with open('nhan.json', 'r') as f:
    labels = json.load(f)
labels = {int(k): v for k, v in labels.items()}  # Chuyển khóa từ string sang int

# Sử dụng bộ phát hiện khuôn mặt của OpenCV (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

# Hàm cảnh báo và lưu trữ thông tin
def canh_bao(frame, timestamp):
    # Tạo thư mục anhcanhbao và vanbancanhbao nếu chưa tồn tại
    """if not os.path.exists("anhcanhbao"):
        os.makedirs("anhcanhbao")
    if not os.path.exists("vanbancanhbao"):
        os.makedirs("vanbancanhbao")"""

    #img_filename_unknow = f"canhbao/anhcanhbao/unknown_{timestamp}.jpg"
    img_filename_unknow = f"Notification/Unknow/img_unknow/unknown_{timestamp}.jpg"
    cv2.imwrite(img_filename_unknow, frame)

    # Tạo file văn bản
    #text_filename_unknow = f"canhbao/vanbancanhbao/{timestamp}.txt"
    text_filename_unknow = f"Notification/Unknow/txt_unknow/{timestamp}.txt"
    with open(text_filename_unknow, "w", encoding='utf-8') as f:
        now = datetime.datetime.now()
        # Định dạng chuỗi thời gian
        formatted_time_unknow = now.strftime("%H h %M m %S s at d %d m %m y %Y")
        # Ghi nội dung vào file
        f.write(f"Phát hiện người lạ lúc {formatted_time_unknow}")

    print(f"Đã chụp ảnh và lưu thông tin vào {img_filename_unknow} và {text_filename_unknow}")

def shipper(frame, timestamp):
    img_filename_shipper = f"Notification/Shipper/img_shipper/shipper_{timestamp}.jpg"
    cv2.imwrite(img_filename_shipper, frame)

    text_filename_shipper = f"Notification/Shipper/txt_shipper/{timestamp}.txt"    
    with open(text_filename_shipper, 'w', encoding='utf-8') as f:
        now = datetime.datetime.now()
        # Định dạng chuỗi thời gian
        formatted_time_shipper = now.strftime("%H h %M m %S s at d %d m %m y %Y")
        # Ghi nội dung vào file
        f.write(f"Phát hiện Shipper lúc {formatted_time_shipper}")

    print(f"Đã chụp ảnh và lưu thông tin vào {img_filename_shipper} và {text_filename_shipper}")    

# Hàm chính để phát hiện và phân loại khuôn mặt
def recognize_faces_from_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở webcam.")
        return

    print("Đang nhận diện khuôn mặt...")

    detected_faces = {}
    last_alert_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ webcam.")
            break

        # Chuyển khung hình sang grayscale để phát hiện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt trong khung hình
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Lấy phần ảnh khuôn mặt
            face_img = frame[y:y+h, x:x+w]

            # Dự đoán nhãn và độ tin cậy
            label, confidence = predict_face(face_img)

            # Chỉ hiển thị nếu nhãn của khuôn mặt này chưa được hiển thị
            if (x, y) not in detected_faces or detected_faces[(x, y)] != label:
                detected_faces[(x, y)] = label

            if confidence < 0.7:
                label = "Unknown"

            label_text = f"{label} ({confidence*100:.2f}%)"
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if label == "Unknown":
                print("Phát hiện khuôn mặt chưa biết")
                current_time = time.time()
                if current_time - last_alert_time >= 10:
                    print("Đã đủ 10 giây, bắt đầu chụp ảnh")
                    canh_bao(frame, timestamp=time.strftime("%Y%m%d_%H%M%S"))
                    last_alert_time = current_time
                else:
                    print("Chưa đủ 10 giây, bỏ qua")

            elif label == "Shipper":
                print("Có shipper")
                current_time = time.time()
                if current_time - last_alert_time >= 10:
                    print("Đã đủ 10 giây, bắt đầu chụp ảnh")
                    shipper(frame, timestamp=time.strftime("%Y%m%d_%H%M%S"))
                    last_alert_time = current_time
                else:
                    print("Chưa đủ 10 giây, bỏ qua")

        # Hiển thị khung hình
        cv2.imshow('Nhận diện khuôn mặt', frame)

        # Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng tất cả cửa sổ
    cap.release()
    cv2.destroyAllWindows()

# Chạy chương trình chính
if __name__ == "__main__":
    recognize_faces_from_webcam()