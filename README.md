# 🌐 PDF Language Classifier

Hệ thống phân loại ngôn ngữ tự động từ file PDF sử dụng XLM-RoBERTa.

## 📊 Dataset

- **Tổng số PDFs:** 9,674 files
- **Ngôn ngữ:**
  - 🇯🇵 Japanese: 3,463 files
  - 🇰🇷 Korean: 1,844 files
  - 🇺🇸 English: 2,447 files
  - 🇻🇳 Vietnamese: 1,920 files

## 🎯 Performance

- **Accuracy:** 98.93%
- **F1-Score:** 0.9893
- **Inference time:** ~0.5s/PDF

## 🛠️ Technology Stack

- **Model:** XLM-RoBERTa Base (560MB)
- **Framework:** PyTorch + Transformers
- **PDF Processing:** PyMupdf
- **UI:** Streamlit
- **Visualization:** Plotly + WandB

## 💻 System Requirements

- **GPU:** NVIDIA RTX 3050 (4GB VRAM)
- **CUDA:** 12.5
- **Python:** 3.10.11
- **RAM:** 16GB
- **Storage:** ~5GB (model + data)

## 📦 Installation

### 1. Clone repository
```bash
git clone <repository-url>
cd pdf_language_classifier
```

### 2. Install dependencies
```bash
# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other packages
pip install pdfminer.six transformers datasets accelerate pandas numpy scikit-learn matplotlib seaborn plotly tqdm streamlit sentencepiece protobuf
```

### 3. Verify CUDA
```bash
python check_cuda.py
```

## 🚀 Quick Start

### 1. Prepare Data
Copy your PDF folders (vn/jp/kr/us) to `data/raw/`:
```
data/raw/
├── vn/  (1,920 PDFs)
├── jp/  (3,463 PDFs)
├── kr/  (1,844 PDFs)
└── us/  (2,447 PDFs)
```

### 2. Process Data
```bash
python src/data_processing.py
```

Time: ~2-3 hours for ~10K PDFs

### 3. Train Model
```bash
python src/train.py
```

Time: ~2-3 hours (3 epochs)

### 4. Run Demo
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

## 📁 Project Structure

```
pdf_language_classifier/
├── data/
│   ├── raw/              # Original PDFs (vn/jp/kr/us)
│   ├── processed/        # Processed data
│   └── processed/splits/ # Train/val/test splits
├── src/
│   ├── data_processing.py  # Data extraction & processing
│   ├── train.py            # Model training
│   └── inference.py        # Inference module
├── models/                 # Trained models
├── app.py                  # Streamlit demo
├── check_cuda.py           # CUDA verification
├── data_analyst.py         # Analyst PDFs data
├── requirements.txt
└── README.md
```

## 🎓 Usage Examples

### Inference from Python

```python
from src.inference import LanguageClassifier

# Load classifier
classifier = LanguageClassifier("models/xlm-roberta-lang-20240110_120000")

# Predict from PDF
result = classifier.predict_from_pdf("test.pdf")
print(f"Language: {result['language']}")
print(f"Confidence: {result['confidence']:.2%}")

# Predict from text
result = classifier.predict_from_text("Your text here")
```

### Batch Processing

```python
pdf_files = ["file1.pdf", "file2.pdf", "file3.pdf"]
results = classifier.batch_predict(pdf_files)
```

## ⚙️ Configuration

### Training Parameters (in `src/train.py`)

```python
CONFIG = {
    'model_name': 'xlm-roberta-base',
    'max_length': 512,              # Giảm xuống 256 nếu OOM
    'epochs': 3,
    'batch_size': 6,                # ⚙️ Giảm từ 8 → 6 (an toàn hơn)
    'gradient_accumulation': 2,     # ⚙️ Effective batch = 12
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'fp16': True,                   # ⚙️ Bật mixed precision
    'max_grad_norm': 1.0,           # ⚙️ Gradient clipping
    'eval_steps': 100,              # ⚙️ Evaluate mỗi 100 steps
    'save_steps': 100,
}
```

## 📊 Model Details

- **Architecture:** XLM-RoBERTa Base
- **Parameters:** 270M
- **Tokenizer:** SentencePiece
- **Max sequence length:** 512 tokens
- **Training time:** ~3-4 hours on RTX 3050
- **Inference time:** ~0.5s per PDF

## 📝 Notes

- Model works best with text-based PDFs
- Scanned PDFs require OCR (not included in this project)
- Minimum text length: 50 characters
- Maximum text processed: 5000 characters per PDF

## 🤝 Contributing

This is a university project. Contributions are welcome!

## 📧 Contact

Created by HUST Student: NGUYEN VIET ANH 

## 📄 License

This project is for educational purposes.

**⭐ If you find this project useful, please give it a star!**