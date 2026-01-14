# 📚 HƯỚNG DẪN THAM SỐ TRAINING

## 🎯 TÓM TẮT NHỮNG THAY ĐỔI QUAN TRỌNG

### ✅ Code mới CẢI THIỆN gì so với code cũ?

| Tính năng | Code cũ | Code mới | Lý do |
|-----------|---------|----------|-------|
| **Batch size** | 8 | 6 | An toàn hơn cho 4GB VRAM |
| **Gradient accumulation** | Không có | 2 | Tăng effective batch lên 12 |
| **Eval strategy** | epoch | steps (100) | Phát hiện overfitting sớm hơn |
| **GPU memory tracking** | Không có | Có | Debug OOM dễ dàng |
| **Confusion matrix** | 1 loại | 2 loại (count + %) | Phân tích tốt hơn |
| **Per-class metrics** | Không chi tiết | Đầy đủ | Biết ngôn ngữ nào yếu |
| **Error handling** | Cơ bản | Chi tiết + gợi ý | Dễ fix lỗi |
| **Config management** | Hardcode | Dict dễ thay đổi | Flexible hơn |

---

## 🔧 GIẢI THÍCH CHI TIẾT CÁC THAM SỐ

### 1️⃣ **BATCH SIZE & GRADIENT ACCUMULATION**

```python
batch_size = 6                    # Số samples/batch thực tế
gradient_accumulation = 2         # Accumulate gradients qua N batches
effective_batch_size = 6 * 2 = 12 # Batch size "cảm nhận" được
```

**📖 Giải thích:**
- **Batch size nhỏ (6)**: Tiết kiệm VRAM, nhưng gradient "noisy" hơn
- **Gradient accumulation (2)**: Tích lũy gradients qua 2 batches trước khi update
- **Kết quả**: Model học như batch_size=12, nhưng chỉ tốn VRAM của batch_size=6

**💡 Khi nào điều chỉnh:**
```python
# Nếu Out of Memory:
batch_size = 4
gradient_accumulation = 3
# Effective batch = 12 (giống như trước)

# Hoặc:
batch_size = 3
gradient_accumulation = 4
# Effective batch = 12
```

**⚠️ Trade-off:**
- ✅ Ưu điểm: Tiết kiệm VRAM, có thể train model lớn
- ❌ Nhược điểm: Chậm hơn (~20-30%) vì phải forward nhiều lần

---

### 2️⃣ **MIXED PRECISION (FP16)**

```python
fp16 = True                       # Bật mixed precision training
fp16_opt_level = "O1"            # O1 = conservative, O2 = aggressive
torch_dtype = torch.float16      # Load model ở FP16
```

**📖 Giải thích:**
- **FP16**: Sử dụng 16-bit floats thay vì 32-bit
- **Tiết kiệm**: ~40-50% VRAM, tăng tốc ~2-3x trên GPU mới
- **O1 (conservative)**: An toàn hơn, giữ một số operations ở FP32
- **O2 (aggressive)**: Nhanh hơn nhưng có thể không ổn định

**💡 Khi nào TẮT FP16:**
```python
fp16 = False
# Dùng khi:
# - Gặp NaN loss
# - Model không converge
# - Accuracy giảm bất thường
```

**⚙️ Cơ chế hoạt động:**
```
Forward pass:  FP16 (nhanh, ít VRAM)
    ↓
Loss:          FP32 (chính xác)
    ↓
Backward:      FP16 (nhanh)
    ↓
Optimizer:     FP32 (ổn định)
```

---

### 3️⃣ **EVALUATION STRATEGY**

```python
# Code cũ:
eval_strategy = "epoch"           # Eval sau mỗi epoch
# Problem: Phát hiện overfitting muộn

# Code mới:
eval_strategy = "steps"           # Eval sau mỗi N steps
eval_steps = 100                  # Mỗi 100 steps
# Benefit: Phát hiện overfitting sớm, dừng kịp thời
```

**📊 So sánh:**

| Strategy | Eval frequency | Use case |
|----------|----------------|----------|
| `epoch` | 1 lần/epoch (~600 steps) | Dataset nhỏ (<1000 samples) |
| `steps` (100) | 6 lần/epoch | Dataset lớn, monitor chặt chẽ |
| `steps` (50) | 12 lần/epoch | Debug, tune hyperparams |

**💡 Tối ưu:**
```python
# Dataset nhỏ (1-2K samples):
eval_steps = 50

# Dataset vừa (5-10K samples):
eval_steps = 100  # ⭐ Đang dùng

# Dataset lớn (50K+ samples):
eval_steps = 500
```

---

### 4️⃣ **LEARNING RATE & WARMUP**

```python
learning_rate = 2e-5              # LR chính
warmup_ratio = 0.1                # 10% steps đầu warmup
warmup_steps = total_steps * 0.1  # Auto calculate
```

**📈 Learning rate schedule:**

```
LR
 ↑
 │     ╱────────────╲
 │    ╱              ╲___
 │   ╱                   ╲___
 │  ╱                        ╲___
 │ ╱                             ╲
 └─────────────────────────────────→ Steps
   ↑                              ↑
   Warmup (10%)                   Decay
```

**📖 Giải thích:**
1. **Warmup phase** (10% đầu): LR tăng dần từ 0 → 2e-5
   - Tránh gradient shock
   - Model ổn định hơn
   
2. **Training phase**: LR = 2e-5 constant
   
3. **Decay phase** (optional): LR giảm dần về 0

**💡 Khi nào điều chỉnh:**

```python
# Model không converge, loss giảm chậm:
learning_rate = 3e-5  # Tăng lên
warmup_ratio = 0.05   # Giảm warmup

# Loss oscillate, không ổn định:
learning_rate = 1e-5  # Giảm xuống
warmup_ratio = 0.15   # Tăng warmup

# Dataset rất nhỏ (<1K samples):
learning_rate = 5e-5  # Tăng mạnh
warmup_ratio = 0.0    # Không cần warmup
```

---

### 5️⃣ **GRADIENT CLIPPING**

```python
max_grad_norm = 1.0               # Clip gradients > 1.0
```

**📖 Giải thích:**
- **Problem**: Đôi khi gradients rất lớn → model explode
- **Solution**: Clip gradients về max_grad_norm
- **Ví dụ**: Nếu gradient = 5.0 → scale về 1.0

**🔢 Cơ chế:**
```python
if gradient_norm > max_grad_norm:
    gradient = gradient * (max_grad_norm / gradient_norm)
```

**💡 Khi nào điều chỉnh:**

```python
# Loss = NaN, model diverge:
max_grad_norm = 0.5   # Clip chặt hơn

# Training ổn định, muốn học nhanh hơn:
max_grad_norm = 2.0   # Cho phép gradient lớn hơn

# Dataset sạch, model ổn định:
max_grad_norm = None  # Không clip
```

---

### 6️⃣ **EARLY STOPPING**

```python
EarlyStoppingCallback(
    early_stopping_patience=3,       # Dừng nếu không improve sau 3 evals
    early_stopping_threshold=0.001   # Cải thiện > 0.001 mới coi là "improve"
)
```

**📊 Hoạt động:**

```
Eval 1: F1 = 0.950 ✅ Best model saved
Eval 2: F1 = 0.951 ✅ Best model saved (+0.001)
Eval 3: F1 = 0.950 ⚠️  No improvement (1/3)
Eval 4: F1 = 0.949 ⚠️  No improvement (2/3)
Eval 5: F1 = 0.948 ⚠️  No improvement (3/3)
        → STOP TRAINING! ⛔
```

**💡 Khi nào điều chỉnh:**

```python
# Dataset nhỏ, train nhanh:
early_stopping_patience = 2   # Dừng sớm hơn

# Dataset lớn, muốn train đủ:
early_stopping_patience = 5   # Kiên nhẫn hơn

# Model vẫn improve chậm nhưng đều:
early_stopping_threshold = 0.0001  # Nhạy hơn với improvement nhỏ
```

---

### 7️⃣ **DATALOADER OPTIMIZATION**

```python
dataloader_num_workers = 2        # Số workers load data
dataloader_pin_memory = True      # Pin memory cho GPU
```

**📖 Giải thích:**

**Workers:**
```
workers=0: CPU load → GPU (chậm)
workers=2: 2 CPUs load song song → GPU (nhanh hơn ~30%)
workers=4: 4 CPUs load song song → GPU (nhanh hơn ~50%)
```

**Pin memory:**
- `True`: Data được pin vào RAM → transfer sang GPU nhanh hơn
- `False`: Data trong RAM thường → transfer chậm hơn

**💡 Tối ưu cho RTX 3050:**

```python
# Máy mạnh (CPU 8+ cores, RAM 16GB+):
dataloader_num_workers = 4
dataloader_pin_memory = True

# Máy trung bình (CPU 4-6 cores, RAM 8-16GB):
dataloader_num_workers = 2  # ⭐ Đang dùng
dataloader_pin_memory = True

# Máy yếu (CPU 2 cores, RAM <8GB):
dataloader_num_workers = 0
dataloader_pin_memory = False
```

---

## 🎯 CẤU HÌNH KHUYẾN NGHỊ

### 📊 **Theo Mức Độ VRAM:**

#### **4GB VRAM (RTX 3050)** - ⭐ CẤU HÌNH HIỆN TẠI
```python
CONFIG = {
    'batch_size': 6,
    'gradient_accumulation': 2,
    'max_length': 512,
    'fp16': True,
    'dataloader_num_workers': 2,
}
# Effective batch: 12
# VRAM usage: ~3.5GB
```

#### **6GB VRAM (RTX 3060)**
```python
CONFIG = {
    'batch_size': 8,
    'gradient_accumulation': 2,
    'max_length': 512,
    'fp16': True,
    'dataloader_num_workers': 4,
}
# Effective batch: 16
# VRAM usage: ~5GB
```

#### **8GB+ VRAM (RTX 3070+)**
```python
CONFIG = {
    'batch_size': 16,
    'gradient_accumulation': 1,
    'max_length': 512,
    'fp16': True,
    'dataloader_num_workers': 4,
}
# Effective batch: 16
# VRAM usage: ~6-7GB
```

---

## 🚨 TROUBLESHOOTING

### ❌ **Out of Memory (OOM)**

**Triệu chứng:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Giải pháp (thử tuần tự):**

1. **Giảm batch size:**
```python
batch_size = 4  # Từ 6 → 4
gradient_accumulation = 3  # Từ 2 → 3
# Effective batch vẫn = 12
```

2. **Giảm max_length:**
```python
max_length = 256  # Từ 512 → 256
# Giảm ~40% VRAM
```

3. **Tắt workers:**
```python
dataloader_num_workers = 0
```

4. **Cuối cùng - tắt FP16:**
```python
fp16 = False
# Chậm hơn ~2x nhưng ổn định
```

### ⚠️ **Training Không Converge**

**Triệu chứng:**
- Loss không giảm sau nhiều epochs
- Accuracy stuck ở ~25% (random)

**Giải pháp:**

1. **Tăng learning rate:**
```python
learning_rate = 3e-5  # Từ 2e-5 → 3e-5
```

2. **Giảm weight decay:**
```python
weight_decay = 0.001  # Từ 0.01 → 0.001
```

3. **Tắt early stopping:**
```python
# Bỏ EarlyStoppingCallback trong callbacks
```

### 🔥 **Loss = NaN**

**Triệu chứng:**
```
Loss: nan
```

**Giải pháp:**

1. **Giảm learning rate:**
```python
learning_rate = 1e-5
```

2. **Clip gradients chặt hơn:**
```python
max_grad_norm = 0.5
```

3. **Tắt FP16:**
```python
fp16 = False
```

---

## 📈 MONITORING TRAINING

### Xem logs real-time:

```bash
# Terminal 1: Training
python src/train_optimized.py

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: TensorBoard (optional)
tensorboard --logdir=models/xlm-roberta-lang-XXXXXX/logs
```

### Metrics cần theo dõi:

```
✅ GOOD SIGNS:
- Loss giảm đều đặn
- Validation F1 tăng
- GPU utilization ~90-100%
- No memory warnings

⚠️ WARNING SIGNS:
- Loss tăng đột ngột → learning rate quá cao
- Val F1 giảm mà train F1 tăng → overfitting
- GPU utilization <50% → bottleneck ở CPU/IO
- Frequent OOM warnings → giảm batch size
```

---

## 🎓 TÓM TẮT

### ⭐ **Top 5 Parameters Quan Trọng Nhất:**

1. **batch_size + gradient_accumulation** 
   - Ảnh hưởng: VRAM usage & training stability
   - Khuyến nghị: `batch_size=6, gradient_accumulation=2`

2. **fp16**
   - Ảnh hưởng: VRAM usage & speed
   - Khuyến nghị: `True` (tiết kiệm 40-50% VRAM)

3. **learning_rate**
   - Ảnh hưởng: Convergence speed & final accuracy
   - Khuyến nghị: `2e-5` (standard cho BERT-like models)

4. **max_length**
   - Ảnh hưởng: VRAM & information capture
   - Khuyến nghị: `512` (giảm xuống 256 nếu OOM)

5. **eval_steps**
   - Ảnh hưởng: Early stopping & overfitting detection
   - Khuyến nghị: `100` (cho 9K dataset)

### 🔄 **Quick Reference:**

```python
# Safe config (chắc chắn chạy được trên RTX 3050 4GB)
SAFE_CONFIG = {
    'batch_size': 4,
    'gradient_accumulation': 3,
    'max_length': 256,
    'fp16': True,
    'learning_rate': 2e-5,
}

# Optimal config (recommended, 95% success rate)
OPTIMAL_CONFIG = {
    'batch_size': 6,         # ⭐ Current
    'gradient_accumulation': 2,
    'max_length': 512,
    'fp16': True,
    'learning_rate': 2e-5,
}

# Aggressive config (fast but risky)
AGGRESSIVE_CONFIG = {
    'batch_size': 8,
    'gradient_accumulation': 1,
    'max_length': 512,
    'fp16': True,
    'learning_rate': 3e-5,
}
```

---

**💡 Lời khuyên cuối:**

1. Bắt đầu với **OPTIMAL_CONFIG**
2. Nếu OOM → chuyển sang **SAFE_CONFIG**
3. Nếu training mượt → thử **AGGRESSIVE_CONFIG**
4. Always monitor GPU memory và training curves!

Good luck! 🚀