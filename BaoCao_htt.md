# BÁO CÁO BÀI TẬP LỚN
## Nhận Diện Biển Báo Giao Thông Sử Dụng Mạng Nơ-ron Sâu

---

## MỤC LỤC

1. [Giới thiệu đề tài](#1-giới-thiệu-đề-tài)
2. [Các nghiên cứu liên quan](#2-các-nghiên-cứu-liên-quan)
3. [Phương pháp đề xuất](#3-phương-pháp-đề-xuất)
   - 3.1. [Dataset và tiền xử lý dữ liệu](#31-dataset-và-tiền-xử-lý-dữ-liệu)
   - 3.2. [Kiến trúc Mạng Nơ-ron Tích Chập (CNN)](#32-kiến-trúc-mạng-nơ-ron-tích-chập-cnn)
   - 3.3. [Spatial Transformer Network (STN)](#33-spatial-transformer-network-stn)
   - 3.4. [Local Contrast Normalization (LCN)](#34-local-contrast-normalization-lcn)
   - 3.5. [Hàm mất mát và tối ưu hóa](#35-hàm-mất-mát-và-tối-ưu-hóa)
4. [Kết quả và thực nghiệm](#4-kết-quả-và-thực-nghiệm)
5. [Kết luận](#5-kết-luận)

---

## 1. Giới Thiệu Đề Tài

Đề tài thực hiện xây dựng hệ thống nhận diện biển báo giao thông (Traffic Sign Recognition System – TSRS) dựa trên phương pháp học sâu (Deep Learning). Hệ thống sử dụng Mạng Nơ-ron Tích Chập (CNN) kết hợp với Spatial Transformer Networks (STN) để phân loại ảnh biển báo giao thông.

Hệ thống nhận diện biển báo giao thông có vai trò quan trọng trong nhiều ứng dụng thực tế như xe tự lái, giám sát giao thông, hỗ trợ lái xe, và quản lý hạ tầng đường bộ. Thách thức chính bao gồm: sự biến đổi về tỷ lệ, góc nhìn, ánh sáng, bị che khuất, màu sắc phai nhạt và mờ do chuyển động.

Dựa trên bài báo của Arcos-García et al. (2017) *"Deep Neural Network for Traffic Sign Recognition Systems: An analysis of Spatial Transformers and Stochastic Optimization Methods"*, đề tài triển khai CNN với 3 Spatial Transformer Network đạt độ chính xác **99.71%** trên bộ dữ liệu GTSRB – vượt qua tất cả các phương pháp trước đó.

---

## 2. Các Nghiên Cứu Liên Quan

Lĩnh vực nhận diện biển báo giao thông đã trải qua các giai đoạn phát triển chính:

- **Phương pháp dựa trên màu sắc và hình dạng**: Sử dụng không gian màu RGB, HIS, HSV để phân đoạn ảnh đường bộ. Các phương pháp phát hiện đối xứng (Loy & Barnes, 2004), Hough transform (Barnes et al., 2010) được áp dụng để nhận dạng hình dạng biển báo.

- **Phương pháp học máy truyền thống**: Kết hợp HOG (Histogram of Oriented Gradients), SVM, Random Forest. Mathias et al. (2013) đạt 98.53% trên GTSRB bằng HOG + INNLP + INNC.

- **Phương pháp học sâu**: Cireşan et al. (2012) đạt 99.46% bằng ủy ban 25 CNN với tăng cường dữ liệu. Sermanet & LeCun (2011) đạt 98.31% với multi-scale CNN. Jin et al. (2014) đạt 99.65% với tập hợp 20 CNN dùng hinge loss SGD.

Hạn chế của các phương pháp trước: cần tăng cường dữ liệu thủ công, sử dụng nhiều CNN song song dẫn đến chi phí bộ nhớ và tính toán cao.

---

## 3. Phương Pháp Đề Xuất

### 3.1. Dataset và Tiền Xử Lý Dữ Liệu

**Bộ dữ liệu GTSRB (German Traffic Sign Recognition Benchmark)**:
- Tập huấn luyện: **39,209 ảnh**
- Tập kiểm tra: **12,630 ảnh**
- Số lớp: **43 loại biển báo**
- Kích thước ảnh gốc: từ 15×15 đến 250×250 pixels (RGB)

**Tiền xử lý**:
- Rescale tất cả ảnh về kích thước **48×48 pixels**
- Chuẩn hóa toàn cục (Global Normalization): trừ mean, chia std
- Chuẩn hóa tương phản cục bộ (Local Contrast Normalization – LCN) với Gaussian kernel

**Trong code `main.py`**:
```python
data_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
```
Chuẩn hóa theo từng kênh màu với mean `(0.3337, 0.3064, 0.3171)` và std `(0.2672, 0.2564, 0.2629)` ở bước tiền xử lý toàn cục. Phép LCN được thực hiện trong hàm `forward()` của model.

---

### 3.2. Kiến Trúc Mạng Nơ-ron Tích Chập (CNN)

Kiến trúc chính của CNN sử dụng trong bài báo (không bao gồm STN):

| Layer | Loại | Số Feature Maps & Neurons | Kernel |
|-------|------|--------------------------|--------|
| 0 | Input | 3 m. of 48×48 n. | — |
| 1 | Convolutional | 200 m. of 46×46 n. | 7×7 |
| 2 | ReLU | 200 m. of 46×46 n. | — |
| 3 | Max-Pooling | 200 m. of 23×23 n. | 2×2 |
| 4 | Local Contrast Norm. | 200 m. of 23×23 n. | — |
| 5 | Convolutional | 250 m. of 24×24 n. | 4×4 |
| 6 | ReLU | 250 m. of 24×24 n. | — |
| 7 | Max-Pooling | 250 m. of 12×12 n. | 2×2 |
| 8 | Local Contrast Norm. | 250 m. of 12×12 n. | — |
| 9 | Convolutional | 350 m. of 13×13 n. | 4×4 |
| 10 | ReLU | 350 m. of 13×13 n. | — |
| 11 | Max-Pooling | 350 m. of 6×6 n. | 2×2 |
| 12 | Local Contrast Norm. | 350 m. of 6×6 n. | — |
| 13 | Fully connected | 400 neurons | 1×1 |
| 14 | ReLU | 400 neurons | — |
| 15 | Fully connected | 43 neurons | 1×1 |
| 16 | Soft-max | 43 neurons | — |

**Tương ứng trong `model.py`**:
```python
class Net(nn.Module):
    def __init__(self):
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 200, kernel_size=7, stride=1, padding=2)   # Layer 1
        self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)             # Layer 3

        self.conv2 = nn.Conv2d(200, 250, kernel_size=4, stride=1, padding=2) # Layer 5
        self.maxpool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)             # Layer 7

        self.conv3 = nn.Conv2d(250, 350, kernel_size=4, stride=1, padding=2) # Layer 9
        self.maxpool3 = nn.MaxPool2d(2, stride=2)                             # Layer 11

        self.FC1 = nn.Linear(12600, 400)  # Layer 13
        self.FC2 = nn.Linear(400, 43)     # Layer 15
```

**Hàm kích hoạt ReLU** (Nair & Hinton, 2010):

$$\text{ReLU}(x) = \max(0, x)$$

```python
x = F.relu(self.conv1(x))   # ReLU sau conv1
x = F.relu(self.conv2(x))   # ReLU sau conv2
x = F.relu(self.conv3(x))   # ReLU sau conv3
y = F.relu(self.FC1(y))     # ReLU sau FC1
```

**Max-Pooling** với kernel 2×2, stride 2: giảm kích thước feature map xuống một nửa.

**Lớp đầu ra** sử dụng `log_softmax`:
```python
return F.log_softmax(y, dim=1)
```

---

### 3.3. Spatial Transformer Network (STN)

STN là module có thể học để thực hiện biến đổi hình học trên feature map, giúp CNN bất biến với phép xoay, dịch chuyển, co giãn và lệch (translation, rotation, scale, skew).

STN gồm 3 thành phần:
1. **Localisation Network** $f_{loc}$: Nhận feature map $U \in \mathbb{R}^{H \times W \times C}$, đầu ra tham số $\theta$ của phép biến đổi
2. **Grid Generator**: Tạo lưới lấy mẫu từ $\theta$
3. **Sampler**: Lấy mẫu bilinear để tạo output $V \in \mathbb{R}^{H' \times W' \times C}$

**Phép biến đổi Affine 2D** (Công thức 1 trong paper):

$$\begin{pmatrix} x_i^s \\ y_i^s \end{pmatrix} = A_\theta \begin{pmatrix} x_i^t \\ y_i^t \\ 1 \end{pmatrix} = \begin{bmatrix} \theta_{11} & \theta_{12} & \theta_{13} \\ \theta_{21} & \theta_{22} & \theta_{23} \end{bmatrix} \begin{pmatrix} x_i^t \\ y_i^t \\ 1 \end{pmatrix}$$

Trong đó $(x_i^t, y_i^t)$ là tọa độ đích trên lưới đầu ra, $(x_i^s, y_i^s)$ là tọa độ nguồn trong feature map đầu vào, và $A_\theta$ là ma trận affine 2×3 được học bởi localisation network.

**Tương ứng trong `model.py`** – ST-1 (trước conv1):
```python
# Localisation network của ST-1
self.st1 = nn.Sequential(
    nn.MaxPool2d(2, stride=2, ceil_mode=True),           # MaxPool: 48x48 -> 24x24
    nn.Conv2d(3, 250, kernel_size=5, stride=1, padding=2), # Conv: 250 maps
    nn.ReLU(True),
    nn.MaxPool2d(2, stride=2, ceil_mode=True),           # MaxPool: 24x24 -> 12x12
    nn.Conv2d(250, 250, kernel_size=5, stride=1, padding=2),
    nn.ReLU(True),
    nn.MaxPool2d(2, stride=2, ceil_mode=True)            # MaxPool: 12x12 -> 6x6
)
self.FC1_ = nn.Sequential(
    nn.Linear(9000, 250),   # 250 * 6 * 6 = 9000
    nn.ReLU(True),
    nn.Linear(250, 6)       # 6 tham số của ma trận affine A_θ
)
```

**Khởi tạo trọng số identity** (để STN ban đầu không biến đổi gì):
```python
self.FC1_[2].weight.data.zero_()
self.FC1_[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
# --> A_θ = [[1,0,0],[0,1,0]] (ma trận identity)
```

**Forward pass của ST-1**:
```python
# ST-1: biến đổi ảnh đầu vào gốc (3 kênh, 48x48)
h1 = self.st1(x)                              # Localisation net
h1 = h1.view(-1, 9000)
h1 = self.FC1_(h1)
theta1 = h1.view(-1, 2, 3)                   # Ma trận affine θ (batch, 2, 3)
grid1 = F.affine_grid(theta1, x.size(),      # Tạo sampling grid
                       align_corners=False)
x = F.grid_sample(x, grid1,                  # Bilinear sampling
                   align_corners=False)
```

**Kiến trúc các localisation network** (theo Table 2 trong paper):

| Layer | ST-1 | ST-2 | ST-3 |
|-------|------|------|------|
| Input | 3 @ 48×48 | 200 @ 23×23 | 250 @ 12×12 |
| MaxPool | 3 @ 24×24 | 200 @ 11×11 | 250 @ 6×6 |
| Conv | 250 @ 24×24 | 150 @ 11×11 | 150 @ 6×6 |
| MaxPool | 250 @ 12×12 | 150 @ 5×5 | 150 @ 3×3 |
| Conv | 250 @ 12×12 | 200 @ 5×5 | 200 @ 3×3 |
| MaxPool | 250 @ 6×6 | 200 @ 2×2 | 200 @ 1×1 |
| FC | 250 neurons | 300 neurons | 300 neurons |
| FC(out) | 6 neurons | 6 neurons | 6 neurons |

Mô hình sử dụng cấu hình tốt nhất **s₁ c s₂ c s₃ c** với 3 STN xen kẽ giữa các khối convolutional.

---

### 3.4. Local Contrast Normalization (LCN)

LCN là phép chuẩn hóa tương phản cục bộ dựa trên Gaussian kernel, giúp tăng cường cạnh và chuẩn hóa mỗi vị trí pixel theo vùng lân cận. Công thức tổng quát:

$$\text{LCN}(x_{ij}) = \frac{x_{ij} - \bar{x}_{ij}}{\max(\sigma_{ij},\ \bar{\sigma})}$$

Trong đó:
- $\bar{x}_{ij}$ là trung bình có trọng số Gaussian của vùng lân cận pixel $(i,j)$
- $\sigma_{ij}$ là độ lệch chuẩn cục bộ của vùng lân cận
- $\bar{\sigma}$ là trung bình của tất cả $\sigma_{ij}$, dùng để tránh chia cho 0

**Gaussian kernel** được định nghĩa trong `model.py`:

$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right), \quad \sigma = 2.0$$

```python
def gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_shape[-1] / 2.)
    for kernel_idx in range(0, kernel_shape[1]):
        for i in range(0, kernel_shape[2]):
            for j in range(0, kernel_shape[3]):
                x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)
    return x / np.sum(x)   # Chuẩn hóa tổng = 1
```

**Hàm LCN** thực tế trong code:

```python
def LCN(image_tensor, gaussian, mid):
    filtered = gaussian(image_tensor)                        # Tính mean Gaussian: x̄_ij
    centered_image = image_tensor - filtered[:, :, mid:-mid, mid:-mid]  # Trừ mean cục bộ
    sum_sqr_XX = gaussian(centered_image.pow(2))             # Tính E[(x - x̄)²]
    denom = sum_sqr_XX[:, :, mid:-mid, mid:-mid].sqrt()     # σ_ij = sqrt(E[(x-x̄)²])
    per_img_mean = denom.mean()                              # σ̄ = mean(σ_ij)
    divisor = denom.clone()
    divisor[per_img_mean > denom] = per_img_mean            # Nếu σ_ij < σ̄ thì dùng σ̄
    divisor[divisor < 1e-4] = 1e-4                          # Tránh chia cho 0
    new_image = centered_image / divisor                     # Chuẩn hóa
    return new_image
```

LCN được gọi 3 lần sau mỗi khối conv-relu-maxpool:
```python
# Sau khối conv1:
mid1 = int(np.floor(self.gfilter1.shape[2] / 2.))
x = LCN(x, self.gaussian1, mid1)

# Sau khối conv2:
mid2 = int(np.floor(self.gfilter2.shape[2] / 2.))
x = LCN(x, self.gaussian2, mid2)

# Sau khối conv3:
mid3 = int(np.floor(self.gfilter3.shape[2] / 2.))
x = LCN(x, self.gaussian3, mid3)
```

Gaussian được triển khai như `nn.Conv2d` với `requires_grad=False` (trọng số cố định, không học):
```python
self.gaussian1 = nn.Conv2d(in_channels=200, out_channels=200,
                            kernel_size=9, padding=8, bias=False)
self.gaussian1.weight.data = self.gfilter1
self.gaussian1.weight.requires_grad = False   # Không cập nhật trong backprop
```

---

### 3.5. Hàm Mất Mát và Tối Ưu Hóa

**Hàm mất mát Cross-Entropy** (Công thức 2 trong paper):

$$H_{y'}(y) = -\sum_i y'_i \log(y_i)$$

Trong đó $y'$ là phân phối xác suất thực (one-hot vector), $y$ là phân phối xác suất dự đoán.

**Hàm Softmax** (Công thức 3 trong paper):

$$f_j(z) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$

Trong code, sử dụng `log_softmax` kết hợp với `nll_loss` tương đương với cross-entropy:
```python
# model.py - output layer:
return F.log_softmax(y, dim=1)

# main.py - loss function:
loss = F.nll_loss(output, target)
# nll_loss(log_softmax(z), y) ≡ cross_entropy(z, y)
```

**Mini-batch Gradient Descent** (Công thức 4 trong paper):

$$w_{k+1} = w_k - \eta_k \nabla \hat{L}(w_k)$$

Trong đó $\nabla \hat{L}(w_k) := \nabla L(w_k; x^{(i:i+n)}; y^{(i:i+n)})$ là gradient trên mini-batch $n$ mẫu, $\eta$ là learning rate.

**Thuật toán tối ưu hóa được chọn: ASGD** (Averaged Stochastic Gradient Descent):

```python
optimizer = optim.ASGD(model.parameters(), lr=args.lr, lambd=0.0001, alpha=0.75,
                        t0=1000000.0, weight_decay=0.0001)
```

**Vòng lặp training**:
```python
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()        # Tính gradient ∇L(w)
    optimizer.step()       # Cập nhật: w ← w - η∇L(w)
```

**Lưu model tốt nhất** theo validation loss nhỏ nhất:
```python
if val_loss < temp:
    temp = val_loss
    model_file = f'model_{epoch}.pth'
    torch.save(model.state_dict(), model_file)
```

---

## 4. Kết Quả và Thực Nghiệm

### 4.1. Thiết Lập Thực Nghiệm

**Tham số huấn luyện trong code của chúng tôi:**

| Tham số | Giá trị |
|---------|---------|
| Batch size | 50 |
| Epochs tối đa | 100 |
| Learning rate | 0.01 |
| Optimizer | ASGD (lambd=0.0001, alpha=0.75, weight\_decay=0.0001) |
| Input size | 48×48 pixels |
| Số lớp | 43 |
| Dataset | GTSRB (torchvision) |

**Tham số thực nghiệm trong paper:**

| Tham số | Giá trị |
|---------|---------|
| Batch size | 50 |
| Epochs thử nghiệm ban đầu | 15 epochs/experiment |
| Epochs cho mô hình tốt nhất | 21 epochs |
| Learning rate (SGD) | 0.01 |
| Optimizer | SGD không momentum |
| Framework | Torch (Lua) |

### 4.2. So Sánh Kết Quả Các Cấu Hình CNN

Bảng kết quả trong paper (15 epochs thử nghiệm):

| Cấu hình CNN | SGD | SGD-N | RMSprop | Adam | # Tham số |
|-------------|-----|-------|---------|------|-----------|
| c c c | 98.31% | 98.33% | 98.66% | 98.81% | 7,303,883 |
| s₁ c c c | 99.09% | 99.15% | 99.37% | 99.20% | 11,137,389 |
| c s₂ c c | 99.22% | 99.13% | 99.28% | 99.15% | 9,046,339 |
| c c s₃ c | 99.02% | 99.04% | 99.11% | 99.39% | 9,053,839 |
| s₁ c s₂ c c | 99.31% | 99.30% | 99.38% | 99.23% | 12,879,845 |
| s₁ c c s₃ c | 99.21% | 99.25% | 99.32% | 99.32% | 12,887,345 |
| c s₂ c s₃ c | 99.34% | 99.23% | 99.45% | 99.28% | 10,796,295 |
| **s₁ c s₂ c s₃ c** | **99.49%** | 99.43% | 99.40% | 99.42% | **14,629,801** |

**Kết quả cuối cùng trên GTSRB** (paper – full training):

| Phương pháp | Accuracy |
|------------|---------|
| **Ours (Single CNN + 3 STNs)** | **99.71%** |
| Jin et al. (2014) – 20 CNN ensemble | 99.65% |
| Cireşan et al. (2012) – 25 CNN committee | 99.46% |
| Yu et al. (2016) – GDBM | 99.34% |
| Human performance (best) | 99.22% |

### 4.3. Phân Tích So Sánh Giữa Code và Paper

**Điểm giống nhau:**
- Cùng kiến trúc CNN 3 khối conv + 3 STN (s₁ c s₂ c s₃ c)
- Cùng kiến trúc localisation network cho mỗi STN
- Cùng sử dụng LCN với Gaussian kernel σ=2.0, kernel 9×9
- Cùng kích thước ảnh đầu vào 48×48
- Cùng batch size 50
- Cùng sử dụng cross-entropy loss (qua log_softmax + nll_loss)

**Điểm khác nhau và lý do thay đổi:**

| Điểm khác biệt | Paper | Code thực thi | Lý do |
|----------------|-------|---------------|-------|
| **Optimizer** | SGD không momentum | **ASGD** (Averaged SGD) | ASGD là biến thể của SGD với averaging trọng số, có lý thuyết hội tụ tốt hơn và ổn định hơn trong thực tế với PyTorch |
| **Framework** | Torch (Lua) | **PyTorch** | PyTorch là framework phổ biến hơn, có hỗ trợ tốt và cộng đồng lớn hơn |
| **Số epoch** | 21 epochs (đạt 99.71%) với 15 epoch thử nghiệm | **100 epochs** tối đa | Tăng epochs để đảm bảo hội tụ đầy đủ; model lưu khi validation loss tốt nhất |
| **Dataset loading** | Custom loader từ file CSV | **torchvision.datasets.GTSRB** | API chính thức của torchvision hỗ trợ GTSRB từ v0.13, đơn giản hóa code |
| **Weight decay** | 1e-4 (SGD) | **0.0001** (ASGD) | Giá trị tương đương, cách viết khác nhau |
| **Validation set** | 12,630 ảnh riêng biệt (GTSRB test set) | Dùng split='test' của GTSRB | Tương đương, torchvision dùng đúng split của benchmark |
| **Demo web** | Không có | **Flask + Web UI** | Thêm giao diện web để demo trực quan với upload ảnh và dự đoán top-5 |

**Tại sao dùng ASGD thay vì SGD thuần?**

ASGD (Averaged Stochastic Gradient Descent) thực hiện trung bình hóa trọng số qua các bước cập nhật:

$$\bar{w}_t = \frac{1}{t - t_0} \sum_{i=t_0}^{t} w_i \quad (t > t_0)$$

Ưu điểm: có bảo đảm lý thuyết về tốc độ hội tụ $O(1/t)$ tốt hơn SGD thường, đặc biệt hiệu quả khi learning rate không giảm. Trong PyTorch, ASGD với `weight_decay=0.0001` mô phỏng gần với khuyến nghị của paper (SGD + weight decay = 1e-4).

### 4.4. Kết Quả Precision, Recall, F1

Kết quả trong paper trên GTSRB:

| Metric | Giá trị |
|--------|---------|
| Precision | 99.71% |
| Recall | 99.71% |
| F1-score | 99.71% |

### 4.5. So Sánh Hiệu Quả Bộ Nhớ

| Phương pháp | Số CNN | Data Augmentation | # Tham số có thể học |
|------------|--------|------------------|---------------------|
| **Ours** | **1** | **Không** | **14,629,801** |
| Jin et al. (2014) | 20 (ensemble) | Có | ~23 triệu |
| Cireşan et al. (2012) | 25 (committee) | Có | ~90 triệu |

Phương pháp đề xuất chỉ sử dụng **1 CNN duy nhất**, không cần tăng cường dữ liệu thủ công, nhưng vẫn đạt kết quả tốt hơn. Nhờ STN học cách biến đổi hình học tự động, mô hình bất biến với các biến đổi không gian mà không cần dữ liệu augmented.

---

## 5. Kết Luận

Đề tài đã triển khai thành công hệ thống nhận diện biển báo giao thông dựa trên kiến trúc CNN với 3 Spatial Transformer Networks (cấu hình s₁ c s₂ c s₃ c) theo bài báo của Arcos-García et al. (2017). Hệ thống bao gồm:

- **Module model.py**: Định nghĩa kiến trúc mạng với 3 STN, 3 khối conv-relu-maxpool-LCN, và 2 lớp fully connected
- **Module main.py**: Thực hiện huấn luyện với ASGD optimizer, lưu model tốt nhất theo validation loss
- **Module app.py**: Giao diện web Flask cho phép upload/chọn ảnh và dự đoán top-5 lớp

Kết quả trong paper đạt **99.71% accuracy** trên GTSRB với 14,629,801 tham số (ít hơn nhiều so với các phương pháp ensemble trước đó). STN giúp mô hình tự động học cách chuẩn hóa không gian, loại bỏ noise hình học mà không cần data augmentation thủ công.

---

*Báo cáo tham khảo: Arcos-García, Á., Álvarez-García, J.A., Soria-Morillo, L.M. (2017). Deep Neural Network for Traffic Sign Recognition Systems: An analysis of Spatial Transformers and Stochastic Optimization Methods. Neural Networks.*
