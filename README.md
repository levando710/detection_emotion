⚙️ Hướng dẫn Cài đặt (Installation) 

Dự án này sử dụng nhiều thư viện xử lý ảnh và Machine Learning. Khuyến khích cài đặt trong môi trường ảo (Virtual Environment) để tránh xung đột thư viện trên máy tính. 

 
Bước 1: Tải mã nguồn về máy (Clone repository) 

Mở Terminal hoặc CMD: tại thư mục bạn muốn lưu dự án (ví dụ: ổ D) và chạy lệnh: 


CMD: 

git clone https://github.com/levando710/detection_emotion.git 

cd detection_emotion 

Bước 2: Tạo và kích hoạt môi trường ảo 

Trên hệ điều hành Windows (sử dụng CMD:): 

 
CMD: 

python -m venv .venv 

.\.venv\Scripts\activate 

Bước 3: Cài đặt các thư viện phụ thuộc 

Đảm bảo bạn đã kích hoạt môi trường ảo (có chữ .venv ở đầu dòng lệnh), sau đó chạy: 

 
CMD: 

pip install -r requirements.txt 

(Lưu ý xử lý lỗi: Nếu quá trình cài đặt báo lỗi tại thư viện dlib, nguyên nhân thường do thiếu trình biên dịch C++ trên Windows. Bạn có thể khắc phục nhanh bằng cách chạy lệnh thay thế: pip install dlib-bin) 


🚀 Hướng dẫn Sử dụng (Usage) 

Hệ thống được chia làm hai giai đoạn chính: Huấn luyện mô hình (Training) và Nhận diện thực tế (Inference). 

 
1. Huấn luyện mô hình từ dữ liệu tĩnh:
2. 
Chạy script dưới đây để hệ thống đọc bộ dữ liệu ảnh, sử dụng thư viện dlib để trích xuất các đặc trưng khuôn mặt (68 landmarks) và tiến hành huấn luyện thuật toán SVM.

 
CMD: 

python train_clf.py (no need)

Kết quả đầu ra: Quá trình huấn luyện diễn ra rất nhanh (vài giây) và sẽ in ra tỷ lệ Accuracy trực tiếp trên màn hình Terminal. 

 
2. Chạy Demo nhận diện qua Webcam:
3. 
Sau khi mô hình đã được huấn luyện thành công, sử dụng script sau để mở luồng Camera trực tiếp. Hệ thống sẽ tự động bắt khuôn mặt và hiển thị nhãn cảm xúc theo thời gian thực.

 
CMD: 

python emotionRecognition.py

Thao tác kết thúc: Để đóng an toàn luồng Camera và kết thúc chương trình, hãy click chuột vào cửa sổ Webcam đang hiển thị và nhấn phím ESC. 

