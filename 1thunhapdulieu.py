import cv2
import os
import shutil

# Hàm để tạo hoặc ghi đè lên thư mục
def create_directory(name):
    base_dir = 'data'  # Thư mục gốc là "data"
    
    # Tạo thư mục gốc nếu chưa tồn tại
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    folder_path = os.path.join(base_dir, name)  # Đường dẫn tới thư mục cần tạo
    
    if os.path.exists(folder_path):
        print(f"Thư mục {folder_path} đã tồn tại.")
        overwrite = input("Bạn có muốn ghi đè lên thư mục này không? (y/n): ").lower()
        if overwrite == 'y':
            shutil.rmtree(folder_path)  # Xóa thư mục hiện tại
            os.makedirs(folder_path)  # Tạo lại thư mục mới
            print(f"Thư mục {folder_path} đã được ghi đè.")
        else:
            # Tạo thư mục mới với số thứ tự đằng sau tên nếu không ghi đè
            i = 1
            while os.path.exists(f"{folder_path}{i}"):
                i += 1
            folder_path = f"{folder_path}{i}"
            os.makedirs(folder_path)
            print(f"Đã tạo thư mục mới: {folder_path}")
    else:
        os.makedirs(folder_path)
        print(f"Đã tạo thư mục: {folder_path}")
    
    return folder_path

# Hàm để xóa thư mục nếu cần
def delete_directory(name):
    folder_path = os.path.join('data', name)  # Xóa từ thư mục gốc 'data'
    
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Đã xóa thư mục: {folder_path}")
    else:
        print(f"Thư mục {folder_path} không tồn tại.")

# Hàm thu thập ảnh từ camera
def collect_images(folder_name):
    cap = cv2.VideoCapture(0)  # Mở camera
    count = 0
    
    print("Nhấn 'c' để thu thập ảnh, 'space' để dừng và lưu.")
    
    while True:
        ret, frame = cap.read()  # Đọc ảnh từ camera
        if not ret:
            print("Không thể truy cập camera.")
            break
        
        cv2.imshow('Camera', frame)  # Hiển thị ảnh từ camera
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            img_path = os.path.join(folder_name, f"image_{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Đã lưu ảnh: {img_path}")
            count += 1
        elif key == ord(' '):  # Phím Space để dừng
            print("Dừng thu thập ảnh.")
            break
    
    cap.release()  # Giải phóng camera
    cv2.destroyAllWindows()  # Đóng tất cả cửa sổ hiển thị

# Chương trình chính
def main():
    while True:
        name = input("Nhập tên của người cần thu thập ảnh hoặc nhập 'xoa' để xoá thư mục: ").strip()
        
        if name == 'xoa':
            folder_to_delete = input("Nhập tên thư mục bạn muốn xóa: ").strip()
            delete_directory(folder_to_delete)
        else:
            folder_name = create_directory(name)  # Tạo thư mục trong 'data'
            collect_images(folder_name)  # Thu thập ảnh vào thư mục đã tạo
        
        repeat = input("Bạn có muốn thu thập thêm dữ liệu không? (y/n): ").lower()
        if repeat != 'y':
            break

if __name__ == "__main__":
    main()
