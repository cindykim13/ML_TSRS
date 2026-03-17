# BÁO CÁO KẾ HOẠCH HÀNH ĐỘNG CHI TIẾT
## TÁI CẤU TRÚC, TỐI ƯU HÓA MÃ NGUỒN VÀ NÂNG CẤP LÝ THUYẾT BÀI BÁO KHOA HỌC

**Mục tiêu:** Dựa trên các lỗ hổng lý thuyết và thực tiễn đã được bóc tách, bản báo cáo này phác thảo lộ trình thực thi (Execution Plan) từng bước để nâng cấp toàn diện dự án. Mục tiêu tối hậu là vượt qua ngưỡng chính xác 99.71% của bài báo gốc, giải quyết triệt để bài toán mất cân bằng dữ liệu, và hiện đại hóa hoàn toàn kiến trúc mạng cũng như thuật toán tối ưu.

---

## GIAI ĐOẠN 1: TÁI CẤU TRÚC VÀ NÂNG CẤP MÃ NGUỒN (CODE REFACTORING & UPGRADE)

Giai đoạn này yêu cầu can thiệp trực tiếp vào các file `.py` hiện tại. Do mã nguồn hiện tại đã làm tốt việc khởi tạo Mạng Biến đổi Không gian (STN Identity Initialization), ta sẽ giữ nguyên phần này và tập trung đại tu các module còn lại.

### Bước 1: Hiện đại hóa Tiền xử lý dữ liệu (`data.py`)

* **Hành động 1.1: Loại bỏ LCN thủ công.** Trong cấu trúc hiện tại, LCN đang được thực hiện như một lớp Convolution (hàm `LCN` và `gaussian_filter` trong `model.py`). Phương pháp này tính toán cồng kềnh và lỗi thời. Cần xóa bỏ hoàn toàn các hàm này khỏi `model.py`.
* **Hành động 1.2: Tích hợp CLAHE và RandAugment.** Cập nhật `data.py` để xử lý nhiễu quang học (photometric invariance) ngay từ khâu load dữ liệu.

**Công việc lập trình:**
* Đưa class `AdaptiveContrastNormalization` (sử dụng OpenCV CLAHE trên kênh YUV) vào `data.py`.
* Cập nhật `data_transforms` cho tập **Train**: Thêm CLAHE → → Tích hợp `torchvision.transforms.v2.RandAugment` → → `ToTensor` → → `Normalize` (giữ nguyên thông số mean/std của GTSRB).
* Cập nhật `data_transforms` cho tập **Test/Validation**: Chỉ dùng Resize → → CLAHE → → `ToTensor` → → `Normalize` (Tuyệt đối không dùng RandAugment cho tập test).

### Bước 2: Nâng cấp Kiến trúc Mạng Trích xuất Đặc trưng (`model.py`)

* **Hành động 2.1: Giữ nguyên module STN.** Giữ lại `self.st1`, `self.st2`, `self.st3` và các đoạn code khởi tạo `FC1_`, `FC2_`, `FC3_` (Identity Initialization) vì chúng đã được triển khai chính xác về mặt toán học.
* **Hành động 2.2: Thay thế Plain CNN bằng Residual + SE-Net.** Xóa bỏ các lớp `self.conv1`, `self.conv2`, `self.conv3` tuần tự hiện tại.

**Công việc lập trình:**
* Khai báo class `SELayer` (Squeeze-and-Excitation) và `ModernResidualConvBlock` ngay trong `model.py`.
* Tái cấu trúc lại luồng `forward(self, x)`: Sau mỗi lần dữ liệu đi qua STN (ví dụ: sau `F.grid_sample`), thay vì đưa vào `F.relu(self.conv1(x))`, hãy đưa dữ liệu qua `ModernResidualConvBlock`.
* > **Lưu ý kỹ thuật đặc biệt:** Khi thay đổi kiến trúc tích chập, số lượng tham số đầu ra (tensor size) khi duỗi phẳng (flatten) để đưa vào `self.FC1` sẽ thay đổi. Cần in ra kích thước tensor `x.shape` ngay trước lớp Fully Connected để cập nhật lại thông số `in_features` của `self.FC1` (hiện tại đang fix cứng là 12600).

### Bước 3: Thay đổi Hàm Mục tiêu để xử lý Mất cân bằng dữ liệu (`main.py`)

* **Hành động 3.1: Loại bỏ Cross Entropy tĩnh.** Tập dữ liệu GTSRB có dạng phân phối đuôi dài. Dùng `F.nll_loss` (Negative Log Likelihood) kết hợp với `log_softmax` ở model hiện tại sẽ khiến các lớp thiểu số bị bỏ qua.

**Công việc lập trình:**
* Định nghĩa class `FocalLoss` trong `main.py` (hoặc tạo file `loss.py` riêng rồi import).
* Trong `model.py`, lớp cuối cùng `self.FC2(y)` KHÔNG áp dụng `F.log_softmax` nữa, chỉ trả về raw logits.
* Trong `main.py` (hàm `train` và `validation`), thay thế `F.nll_loss` bằng đối tượng `criterion = FocalLoss(gamma=2.0, alpha=0.25)`.

### Bước 4: Tích hợp Chiến lược Tối ưu hóa Siêu hội tụ (`main.py`)

* **Hành động 4.1: Thay thế Optimizer lạc hậu.** Hàm `optim.ASGD` hiện tại là một giải pháp chắp vá và cực kỳ chậm trong việc thoát khỏi điểm yên ngựa.

**Công việc lập trình:**
* Đổi optimizer thành: `optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)`. (Sử dụng Decoupled Weight Decay).
* Khởi tạo lịch trình học: `scheduler = torch.optim.lr_scheduler.OneCycleLR(...)` với `max_lr=5e-3` và tổng số steps tính toán dựa trên `epochs` và `len(train_loader)`.
* Trong hàm `train`, THÊM `nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` NGAY TRƯỚC `optimizer.step()` để bảo vệ STN khỏi bùng nổ gradient.
* THÊM `scheduler.step()` NGAY SAU `optimizer.step()` (lưu ý: OneCycleLR cập nhật theo từng batch, không phải từng epoch).

---

## GIAI ĐOẠN 2: TÁI CẤU TRÚC VÀ VIẾT LẠI BÁO CÁO KHOA HỌC (PAPER RESTRUCTURING)

Sau khi hoàn thành phần Code, bạn cần phản ánh sự vượt trội của phương pháp mới vào tài liệu báo cáo học thuật (Đồ án của bạn). Cấu trúc báo cáo cần được viết lại theo trình tự sau:

### Mục 1: Đặt vấn đề và Phê phán Lý thuyết gốc (Critique of Original Work)
* **Trình bày:** Đưa các phân tích từ `Bao_Cao_Phan_Bien_Paper` vào. Dẫn chứng rõ ràng việc tác giả gốc (Álvaro Arcos-García) đã sai lầm khi cho rằng STN có thể thay thế hoàn toàn Data Augmentation (nhầm lẫn giữa hình học và quang học).
* **Trình bày:** Chỉ ra sự thiếu sót trong việc sử dụng Plain CNN và SGD tĩnh, cũng như việc tác giả gốc bỏ qua vấn đề class imbalance của GTSRB.

### Mục 2: Phương pháp luận Hiện đại hóa (Proposed Modern Methodology)
Chia làm 3 tiểu mục tương ứng với 3 trụ cột đã nâng cấp:
* **Tiền xử lý và Tăng cường dữ liệu:** Trình bày lý thuyết về CLAHE và RandAugment. Đưa công thức Focal Loss vào và giải thích cơ chế down-weighting các mẫu dễ (easy examples).
* **Kiến trúc lai (Hybrid Architecture):** Vẽ lại sơ đồ mạng mới. Giải thích cách STN kết hợp với Residual Blocks và SE-Net. Nhấn mạnh việc SE-Net cung cấp Channel Attention, bù đắp cho Spatial Attention của STN.
* **Chiến lược Tối ưu hóa (Optimization Strategy):** Trích dẫn bài báo của Loshchilov & Hutter (2019) về AdamW và bài báo của Leslie N. Smith (2018) về Super-Convergence (OneCycleLR). Giải thích lý do tại sao bộ đôi này vượt trội hơn SGD tĩnh.

### Mục 3: Thực nghiệm và Đánh giá (Experiments & Results)
* **Thiết lập Baseline:** Trình bày kết quả của đoạn code GitHub mà bạn vừa chạy (có STN Identity Init, dùng ASGD, LCN thủ công) làm điểm chuẩn (Baseline).
* **So sánh hiệu suất:** So sánh Baseline với Model đã nâng cấp. Báo cáo cần có bảng biểu so sánh:
    * Độ chính xác tổng thể (Accuracy).
    * Điểm F1-Score trên các lớp thiểu số (chứng minh tác dụng của Focal Loss).
    * Tốc độ hội tụ (Vẽ biểu đồ Loss Curve chứng minh OneCycleLR giúp hàm mất mát giảm nhanh và mượt hơn ở các epoch giữa).

---

## GIAI ĐOẠN 3: TRIỂN KHAI THỰC NGHIỆM VÀ XÁC THỰC (EXECUTION & VALIDATION)

Để đảm bảo quy trình khoa học được tuân thủ nghiêm ngặt, hãy thực thi theo thứ tự sau:

* **Khóa Seed (Determinism):** Đảm bảo `torch.manual_seed(args.seed)` trong `main.py` luôn được kích hoạt. Thêm cấu hình `torch.backends.cudnn.deterministic = True` để kết quả có thể tái lập (reproducible) 100%.
* **Huấn luyện Baseline:** Chạy file `main.py` từ mã nguồn gốc mà bạn cung cấp. Lưu lại các file weights (ví dụ: `model_baseline.pth`) và log lại Training/Validation Loss, Accuracy.
* **Huấn luyện Mô hình Đề xuất (Proposed Model):** Áp dụng toàn bộ các thay đổi lập trình ở Giai đoạn 1. Chạy lại huấn luyện. Lưu lại file weights mới.
* **Đánh giá Độc lập:** Sử dụng file `evaluate2.py` để test cả hai mô hình trên tập Test Dataset độc lập (nếu có Kaggle test set). Mức độ chính xác của Mô hình Đề xuất phải đạt > 99.71% và tiệm cận 99.85% - 99.90% để chứng minh tính siêu việt.