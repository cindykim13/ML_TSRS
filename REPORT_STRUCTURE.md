# PHẦN MỞ ĐẦU (Trang bìa, Mục lục, Danh mục)

**Hướng dẫn:** Phần này tuân thủ tuyệt đối format của trường đại học. Cần chú ý lập Danh mục từ viết tắt đầy đủ cho các thuật ngữ chuyên ngành (ví dụ: CNN, STN, LCN, CLAHE, SE-Net, SGDR, AdamW, GTSRB). Đối với Danh mục hình vẽ và bảng biểu, cần có tên rõ ràng mang tính học thuật (ví dụ: "Bảng 4.1: So sánh hiệu suất định lượng giữa mô hình gốc và kiến trúc SE-ResNet đề xuất").

# CHƯƠNG 1. GIỚI THIỆU
Chương này đặt vấn đề và vạch ra chiến lược giải quyết của toàn bộ đồ án.

## 1.1. Tổng quan bài toán:
* Trình bày bối cảnh của bài toán Thị giác Máy tính trong nhận dạng biển báo giao thông (TSRS).
* Giới thiệu bài báo gốc của Álvaro Arcos-García (2017) và tập dữ liệu GTSRB. Nêu bật thành tựu của bài báo tại thời điểm đó (đạt 99.71% nhờ STN và CNN).
* Giới thiệu mã nguồn GitHub được sử dụng làm cơ sở tái tạo.

## 1.2. Hạn chế của các phương pháp hiện tại:
* Nêu rõ sự chênh lệch giữa học thuật và thực hành: Mã nguồn triển khai GitHub tồn tại các lỗi nghiêm trọng về khởi tạo trọng số và tiền xử lý dữ liệu.
* Nêu rõ sự lỗi thời của lý thuyết: Kiến trúc Plain CNN, việc bỏ qua Data Augmentation, không xử lý mất cân bằng dữ liệu, và việc sử dụng thuật toán tối ưu hóa tĩnh.

## 1.3. Phương pháp đề xuất:
Trình bày chiến lược thực hiện đồ án theo Lộ trình 2 Giai đoạn:
* **Giai đoạn 1 (Fidelity Correction):** Sửa lỗi mã nguồn gốc để tái lập tính trung thực của lý thuyết bài báo.
* **Giai đoạn 2 (Theoretical Transcendence):** Áp dụng các kỹ thuật Học sâu hiện đại để phá vỡ giới hạn của bài báo gốc.

## 1.4. Đóng góp của đề tài:
* Khẳng định 3 đóng góp cốt lõi: (1) Chỉ ra lỗ hổng khởi tạo STN trong các bản tái tạo mã nguồn hiện hành; (2) Tích hợp thành công cơ chế Squeeze-and-Excitation và Focal Loss; (3) Cung cấp mã nguồn Pytorch tối ưu hóa đạt hiệu suất vượt ngưỡng 99.71% của tác giả gốc.

# CHƯƠNG 2. CÁC NGHIÊN CỨU LIÊN QUAN
Chương này cung cấp nền tảng lý thuyết vững chắc cho những chỉ trích và cải tiến của bạn.

## 2.1. Kiến trúc nhận dạng cơ sở (Theo bài báo gốc):
* Trình bày toán học cơ bản của Mạng Biến đổi Không gian (STN): Mạng định vị, Lưới lấy mẫu, Phép biến đổi Affine.
* Trình bày phương pháp Local Contrast Normalization (LCN) bằng Gaussian Filter.

## 2.2. Các tiến bộ trong Học sâu hiện đại (Cơ sở cho cải tiến):
* Trình bày lý thuyết về CLAHE trong tiền xử lý ảnh.
* Trình bày cơ chế Attention (SE-Net) và Mạng thặng dư (Residual Networks) để khắc phục triệt tiêu gradient.
* Trình bày cơ sở toán học của AdamW (Decoupled Weight Decay) và OneCycleLR (Super-Convergence).

# CHƯƠNG 3. PHƯƠNG PHÁP ĐỀ XUẤT (TRỌNG TÂM ĐỒ ÁN)

> **Lưu ý:** Chương này BẮT BUỘC sử dụng cấu trúc "Điểm lý thuyết -> Thực tế mã nguồn -> Vấn đề/Lỗ hổng -> Đề xuất khắc phục (kèm code)" theo đúng chỉ thị.
Bạn chia chương này thành 2 tiểu mục lớn tương ứng với 2 giai đoạn đã vạch ra.

## 3.1. Giai đoạn 1: Hiệu chỉnh tính trung thực của mã nguồn (Fidelity Correction)

### 3.1.1. Khởi tạo Mạng Biến đổi Không gian (STN Initialization)
* **Điểm lý thuyết:** Mạng định vị cần xuất ra tham số θ θ để thực hiện phép biến đổi affine. Tại epoch đầu tiên, mô hình cần thực hiện phép biến đổi đồng nhất (identity mapping).
* **Thực tế mã nguồn:** Hầu hết các mã nguồn tái tạo trên GitHub khởi tạo lớp tuyến tính cuối cùng một cách ngẫu nhiên. (Ghi chú: Mã nguồn mới nhất bạn cung cấp đã cập nhật điều này, nhưng bạn vẫn phải ghi nhận đây là bước hiệu chỉnh đầu tiên mà bạn đã rà soát và xác nhận).
* **Vấn đề/Lỗ hổng:** Khởi tạo ngẫu nhiên gây biến dạng ảnh hỗn loạn, phá hủy lan truyền thuận, khiến gradient bất ổn và mạng rơi vào cực tiểu cục bộ.
* **Đề xuất khắc phục:** Lớp hồi quy cuối cùng BẮT BUỘC phải khởi tạo trọng số bằng 0 và bias là [1, 0, 0, 0, 1, 0]. (Chèn đoạn code khởi tạo `self.FC1_[2].weight.data.zero_()` vào đây).

### 3.1.2. Tiền xử lý dữ liệu và Chuẩn hóa tương phản
* **Điểm lý thuyết:** Bài báo yêu cầu chuẩn hóa tương phản cục bộ (LCN) để xử lý ánh sáng cực đoan.
* **Thực tế mã nguồn:** Mã nguồn gốc (GitHub cũ) chỉ dùng Global Normalization. Mã nguồn hiện tại (bạn mới cung cấp) đã thêm hàm `gaussian_filter` và `LCN` tự định nghĩa.
* **Vấn đề/Lỗ hổng:** Việc tự định nghĩa LCN bằng Gaussian kernel thủ công tiêu tốn tài nguyên tính toán trong quá trình forward pass và không thực sự tối ưu cho viền ảnh bằng các thuật toán xử lý ảnh chuyên dụng.
* **Đề xuất khắc phục:** Thay thế hoàn toàn hàm LCN thủ công bằng thuật toán CLAHE thực thi ở bước `transforms` của Dataset. (Chèn đoạn code `AdaptiveContrastNormalization` sử dụng OpenCV CLAHE không gian YUV vào đây).

### 3.1.3. Chiến lược Tối ưu hóa tĩnh
* **Điểm lý thuyết:** Bài báo sử dụng SGD tĩnh với LR=0.01.
* **Thực tế mã nguồn:** Sử dụng `optim.ASGD` với learning rate tĩnh, không có scheduler, không có Gradient Clipping.
* **Vấn đề/Lỗ hổng:** Việc thiếu bộ lịch trình học (LR scheduler) khiến mạng không thể hội tụ sâu tại các minimum rộng (flat minima).
* **Đề xuất khắc phục:** Sử dụng thuật toán SGDR thông qua `CosineAnnealingWarmRestarts` kết hợp `nn.utils.clip_grad_norm_`. (Chèn đoạn code khởi tạo Scheduler và Training loop có clip_grad_norm vào đây).

## 3.2. Giai đoạn 2: Nâng cấp kiến trúc và Vượt trội lý thuyết (Theoretical Transcendence)

### 3.2.1. Chống Mất cân bằng lớp và Tăng cường quang học
* **Điểm lý thuyết:** Tác giả khẳng định STN thay thế được Data Augmentation. Bài báo dùng Cross-Entropy trên dữ liệu mất cân bằng.
* **Thực tế mã nguồn / Bài báo:** Bỏ qua hoàn toàn biến dạng quang học và sự chênh lệch số lượng mẫu giữa các lớp.
* **Vấn đề/Lỗ hổng:** STN chỉ giải quyết bất biến hình học, không giải quyết bất biến quang học (độ tương phản, sương mù). Cross-Entropy khiến mô hình thiên lệch về các lớp đa số.
* **Đề xuất khắc phục:** Bổ sung `RandAugment` và thay thế hàm mất mát bằng `Focal Loss`. (Chèn đoạn code `modern_augmentation` và class `FocalLoss` vào đây).

### 3.2.2. Hiện đại hóa Kiến trúc Trích xuất đặc trưng
* **Điểm lý thuyết:** Bài báo sử dụng Plain CNN (Conv -> ReLU -> MaxPool).
* **Thực tế mã nguồn / Bài báo:** Không có kết nối thặng dư (Residual), không có cơ chế chú ý (Attention).
* **Vấn đề/Lỗ hổng:** Plain CNN có nguy cơ triệt tiêu gradient khi mạng sâu hơn và chỉ trích xuất đặc trưng không gian, bỏ qua tương quan kênh (channel correlation).
* **Đề xuất khắc phục:** Thay thế các khối tích chập cơ bản bằng `ModernResidualConvBlock` tích hợp `SELayer`. (Chèn đoạn code lớp `SELayer` và `ModernResidualConvBlock` vào đây).

### 3.2.3. Cập nhật Bộ tối ưu hóa SOTA
* **Điểm lý thuyết:** Tác giả kết luận SGD tổng quát hóa tốt hơn Adam.
* **Thực tế mã nguồn / Bài báo:** Loại bỏ hoàn toàn các bộ tối ưu hóa thích ứng.
* **Vấn đề/Lỗ hổng:** Kết luận này là sai lầm do Adam truyền thống xử lý Weight Decay sai lệch. SGD tĩnh hội tụ chậm.
* **Đề xuất khắc phục:** Sử dụng `AdamW` kết hợp với chính sách Super-Convergence `OneCycleLR`. (Chèn đoạn code cấu hình `AdamW` và `OneCycleLR` vào đây).

# CHƯƠNG 4. KẾT QUẢ VÀ THỰC NGHIỆM
Chương này phải chứng minh bằng số liệu minh chứng khoa học (empirical evidence) rằng các cải tiến ở Chương 3 thực sự mang lại hiệu quả.

## 4.1. Các tập dataset:
* Trình bày thông số tập GTSRB (39,209 ảnh train, 12,630 ảnh test, 43 classes).
* Chèn biểu đồ phân phối dữ liệu (Histogram) để minh họa rõ sự mất cân bằng dữ liệu cực cao.

## 4.2. Kết quả:

### 4.2.1. So sánh định lượng:
* Tạo Bảng so sánh 3 mô hình: (1) Baseline (Mã nguồn GitHub cũ), (2) Tái tạo lý thuyết (Đã fix STN + LCN), (3) Mô hình Đề xuất (SE-ResNet + AdamW + Focal Loss).
* **Các chỉ số cần có:** Accuracy (%), Precision, Recall, F1-Score (để chứng minh Focal Loss hoạt động hiệu quả trên lớp thiểu số), Số lượng tham số (Parameters), Thời gian hội tụ (Epochs to converge).
* **Phân tích:** Nhấn mạnh việc mô hình đề xuất phá vỡ kỷ lục 99.71% của bài báo.

### 4.2.2. So sánh định tính:
* **Biểu đồ Loss/Accuracy:** So sánh đường cong hội tụ. Chỉ ra sự sụt giảm Loss đột ngột và mượt mà của mô hình dùng OneCycleLR so với sự dao động của SGD tĩnh.
* **Biểu diễn STN:** Trích xuất và trực quan hóa các khung ảnh trước và sau khi đi qua STN để chứng minh STN đã định vị thành công vùng chứa biển báo.
* **Confusion Matrix:** Đưa ra ma trận nhầm lẫn để phân tích xem mô hình còn sai sót ở những loại biển báo nào mang tính tương đồng quang học cao.

# CHƯƠNG 5. KẾT LUẬN

* **Tổng kết:** Khẳng định lại việc đã hoàn thành nhiệm vụ kép: Vừa tái tạo/hiệu chỉnh thành công lý thuyết của bài báo năm 2017, vừa chứng minh năng lực nâng cấp hệ thống bằng các công nghệ Học sâu hiện đại (SE-Net, AdamW, Focal Loss).
* **Hướng phát triển:** Đề xuất các nghiên cứu trong tương lai, chẳng hạn như tối ưu hóa mô hình sang định dạng TensorRT để triển khai trên các hệ thống biên (Edge Devices) trên xe tự lái, hoặc nghiên cứu tính bền vững của mô hình trước các cuộc tấn công đối kháng (Adversarial Attacks).