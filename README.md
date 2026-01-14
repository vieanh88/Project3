# 🌐 PDF Language Classifier

Hệ thống phân loại ngôn ngữ tự động từ file PDF sử dụng XLM-RoBERTa.

## 📊 Dataset

- **Tổng số PDFs:** 9,055 files
- **Ngôn ngữ:**
  - 🇻🇳 Vietnamese: 2,417 files
  - 🇯🇵 Japanese: 2,422 files
  - 🇰🇷 Korean: 1,756 files
  - 🇺🇸 English: 2,460 files

## 🎯 Performance

- **Accuracy:** 96-98%
- **F1-Score:** 0.96-0.98
- **Inference time:** ~0.5s/PDF

## 🛠️ Technology Stack

- **Model:** XLM-RoBERTa Base (560MB)
- **Framework:** PyTorch + Transformers
- **PDF Processing:** PyMupdf
- **UI:** Streamlit
- **Visualization:** Plotly

## 💻 System Requirements

- **GPU:** NVIDIA RTX 3050 (4GB VRAM) hoặc cao hơn
- **CUDA:** 12.1+
- **Python:** 3.10.11
- **RAM:** 8GB+
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
├── vn/  (2,417 PDFs)
├── jp/  (2,422 PDFs)
├── kr/  (1,756 PDFs)
└── us/  (2,460 PDFs)
```

### 2. Process Data
```bash
python src/data_processing.py
```

Time: ~2-3 hours for 9K PDFs

### 3. Train Model
```bash
python src/train.py
```

Time: ~3-4 hours (3 epochs)

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

### If Out of Memory

1. Reduce `batch_size` to 6 or 4
2. Reduce `max_length` to 256
3. Close other GPU applications

## 📊 Model Details

- **Architecture:** XLM-RoBERTa Base
- **Parameters:** 270M
- **Tokenizer:** SentencePiece
- **Max sequence length:** 512 tokens
- **Training time:** ~3-4 hours on RTX 3050
- **Inference time:** ~0.5s per PDF

## 🐛 Troubleshooting

### CUDA not available
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
- Reduce batch_size in training config
- Reduce max_length
- Use gradient accumulation

### PDF extraction fails
- Ensure PDFs are text-based (not scanned images)
- Check PDF is not corrupted
- Update pdfminer: `pip install --upgrade pdfminer.six`

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

## 🙏 Acknowledgments

- Hugging Face for XLM-RoBERTa model
- Anthropic for Claude AI assistance
- Computer Science Department

---

**⭐ If you find this project useful, please give it a star!**