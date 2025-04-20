# Nhận dạng Tiền giấy Việt Nam bằng mô hình YOLOv5 (Banknote Recognition using YOLOv5)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/drive/folders/1uq-Zw1yGAM6skoff1dgk4c3V1n6gECgQ?usp=sharing)
Dự án này sử dụng mô hình YOLOv5 để phát hiện và nhận dạng các mệnh giá tiền giấy Việt Nam. Hướng dẫn này tập trung vào việc chạy dự án bằng Google Colaboratory.

## Giới thiệu

Mục tiêu của dự án là xây dựng một hệ thống có khả năng xác định các loại tiền giấy khác nhau xuất hiện trong ảnh hoặc luồng video. Mô hình được huấn luyện trên một tập dữ liệu tùy chỉnh gồm hình ảnh các loại tiền giấy Việt Nam.

## Thiết lập trên Google Colab

1.  **Mở Notebook trong Colab:** Nhấn vào huy hiệu "Open In Colab" ở trên để mở tệp notebook chính trong Google Colab.
2.  **Chọn Runtime có GPU:** Để tăng tốc độ huấn luyện và phát hiện, hãy đảm bảo bạn đã chọn môi trường thực thi có GPU: Vào menu `Runtime` -> `Change runtime type` -> Chọn `GPU` trong phần "Hardware accelerator".
3.  **Chạy các ô cài đặt đầu tiên trong Notebook:** Các ô mã dưới đây (hoặc tương đương) nên có ở phần đầu notebook của bạn để thiết lập môi trường.

    * **Kết nối Google Drive:**
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```
    
    * **Clone Repository từ GitHub:**
        ```python
        # Clone kho chứa này vào môi trường Colab
        !git clone [https://github.com/Hng-dev-guy/Banknotes-Recognition.git](https://github.com/Hung-dev-guy/Banknotes-Recognition.git)
        %cd Banknotes-Recognition
        ```


    * **Cài đặt Thư viện:**
        ```python
        # Cài đặt các thư viện chính:
        !pip install torch torchvision torchaudio opencv-python-headless matplotlib numpy PyYAML tqdm pandas seaborn ultralytics -q
        ```

## Chuẩn bị Tập dữ liệu (Dataset)

1.  **Tải/Chuẩn bị Dataset:**
    * Dự án này yêu cầu một tập dữ liệu hình ảnh tiền giấy đã được gán nhãn theo định dạng YOLO.
    * Bạn cần có sẵn thư mục dataset này (ví dụ: thư mục có tên `Dataset_2` như bạn đã dùng) trên **Google Drive** của bạn.
    * **Lưu ý:** Do kích thước lớn, tập dữ liệu không được bao gồm trong kho chứa GitHub này.

2.  **Cập nhật Tệp Cấu hình Dataset (`data.yaml`):**
    * Sau khi clone repository ở bước trên, bạn cần **chỉnh sửa tệp cấu hình YAML** (ví dụ: `Dataset_2/data.yaml` hoặc một tệp tương tự nằm trong thư mục `Banknotes-Recognition` vừa clone về Colab) để nó trỏ đúng đến vị trí dataset trên **Google Drive CỦA BẠN**.
    * Mở tệp YAML đó trong Colab (dùng trình soạn thảo file bên trái hoặc lệnh `!nano path/to/data.yaml`).
    * Đảm bảo các đường dẫn `train:` và `val:` là **đường dẫn tuyệt đối** trỏ vào thư mục dataset trên Drive của bạn.
    * **Ví dụ nội dung cần sửa trong `data.yaml`:**
      ```yaml
      # Ví dụ đường dẫn tuyệt đối trong Colab sau khi mount Drive
      train: /content/drive/MyDrive/path/to/your/Dataset_2/images/train # <<< SỬA ĐƯỜNG DẪN NÀY
      val: /content/drive/MyDrive/path/to/your/Dataset_2/images/val   # <<< SỬA ĐƯỜNG DẪN NÀY

      nc: SO_LUONG_LOP # Số lượng lớp (mệnh giá tiền)
      names: ['TEN_LOP_1', 'TEN_LOP_2', ...] # Danh sách tên các lớp
      ```
    * Thay `/path/to/your/Dataset_2/` bằng đường dẫn thực tế trên Google Drive của bạn.

## Sử dụng (Chạy trong Colab)

Sau khi cài đặt và cấu hình dataset, bạn có thể chạy các tác vụ chính bằng các lệnh trong các ô mã Colab:

*(Giả định các script train.py, val.py, detect.py nằm trong thư mục con như `yolov5/` hoặc `yolov5m/` bên trong `Banknotes-Recognition`. Hãy điều chỉnh đường dẫn đến script cho đúng)*

1.  **Huấn luyện (Training):**
    *(Cung cấp lệnh huấn luyện ví dụ. Đường dẫn `--data` trỏ tới tệp yaml bạn vừa sửa. Đường dẫn `--weights` có thể là trọng số gốc yolov5m.pt hoặc bỏ qua để huấn luyện từ đầu. Kết quả sẽ lưu vào thư mục `runs/train/...`)*
    ```bash
    !python yolov5m/train.py --img 640 --batch 16 --epochs 100 --data Dataset_2/data.yaml --weights yolov5m.pt --cfg yolov5m/models/yolov5m.yaml --name ten_ket_qua_train --project /content/drive/MyDrive/Colab_Results # Lưu kết quả vào Drive
    ```

2.  **Đánh giá (Evaluation):**
    *(Đánh giá mô hình đã huấn luyện. Đường dẫn `--weights` trỏ tới tệp `best.pt` bạn muốn đánh giá, có thể nằm trên Drive hoặc trong thư mục `runs/...` của Colab. Đường dẫn `--data` trỏ tới tệp yaml đã sửa).*
    ```bash
    !python yolov5/val.py --weights /content/drive/MyDrive/Hung/Banknote_Recognition/yolov5m/train/exp6/weights/best.pt --data Dataset_2/data.yaml --imgsz 640 --conf-thres 0.001 --iou-thres 0.6 --task val --project /content/drive/MyDrive/Colab_Results/Eval # Lưu kết quả vào Drive
    ```

3.  **Phát hiện (Detection):**
    *(Chạy phát hiện trên ảnh/video. Đường dẫn `--weights` như trên. `--source` là đường dẫn đến ảnh/video trên Drive hoặc Colab, hoặc `0` cho webcam nếu Colab hỗ trợ).*
    ```bash
    !python yolov5m/detect.py --weights /content/drive/MyDrive/Hung/Banknote_Recognition/yolov5m/train/exp6/weights/best.pt --imgsz 640 --conf-thres 0.5 --source /content/drive/MyDrive/path/to/image.jpg --project /content/drive/MyDrive/Colab_Results/Detect # Lưu kết quả vào Drive
    ```

## Mô hình

* Mô hình được sử dụng trong dự án này dựa trên kiến trúc **YOLOv5**.
* Trọng số được huấn luyện tốt nhất (`best.pt`) mà tác giả sử dụng có thể tìm thấy tại `/content/drive/MyDrive/Hung/Banknote_Recognition/yolov5/runs/train/exp3/weights/best.pt` (nếu bạn muốn chia sẻ đường dẫn này, nhưng thường người dùng sẽ tự huấn luyện hoặc dùng trọng số họ có). *(Cân nhắc việc cung cấp sẵn trọng số `best.pt` trong repo hoặc trên Drive nếu bạn muốn người khác dễ dàng sử dụng ngay)*

## Lưu ý cho người dùng Colab

* **Thời gian chạy:** Vì các phiên Colab miễn phí có giới hạn thời gian, nên quá trình huấn luyện dài có thể bị ngắt. Cân nhắc sử dụng Colab Pro hoặc lưu lại checkpoints thường xuyên vào Google Drive (`--project /content/drive/MyDrive/...`).
* **Đường dẫn:** Người dùng cần thay đổi đường dẫn trong notebook để customize trên môi trường colab cá nhân

