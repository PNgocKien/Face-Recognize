import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# Đường dẫn đến thư mục data chứa ảnh
DATA_DIR = 'data'

# Kích thước ảnh cố định để đưa vào mạng CNN
IMG_SIZE = (64, 64)

# Tạo dữ liệu từ thư mục chứa ảnh
def load_data(data_dir):
    X = []
    y = []
    labels = {}

    label_index = 0
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        
        if os.path.isdir(label_path):
            labels[label_index] = label
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path)

                if img is not None:
                    # Resize ảnh về kích thước cố định
                    img = cv2.resize(img, IMG_SIZE)
                    # Chuyển ảnh về dạng array và thêm vào tập dữ liệu
                    X.append(img)
                    y.append(label_index)
            label_index += 1

    # Chuyển đổi dữ liệu về dạng NumPy array
    X = np.array(X)
    y = np.array(y)

    # Chuẩn hóa dữ liệu ảnh
    X = X / 255.0

    return X, y, labels

# Xây dựng mô hình CNN
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile mô hình
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # Tải dữ liệu
    print("Đang tải dữ liệu...")
    X, y, labels = load_data(DATA_DIR)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuyển đổi nhãn thành dạng one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Xây dựng mô hình
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)  # 3 kênh màu (RGB)
    num_classes = len(labels)

    print("Đang xây dựng mô hình...")
    model = build_model(input_shape, num_classes)

    # Huấn luyện mô hình với EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Đang huấn luyện mô hình...")
    model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Lưu mô hình đã huấn luyện
    model.save('Mohinh_nhandien.h5')
    print("Đã lưu mô hình vào file Mohinh_nhandien.h5")

    # Lưu nhãn lớp vào file nhan.json
    with open('nhan.json', 'w') as f:
        json.dump(labels, f)
    print("Đã lưu nhãn lớp vào file nhan.json")