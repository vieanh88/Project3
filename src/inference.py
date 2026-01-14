"""
Module inference cho PDF Language Classifier
Author: Nguyễn Việt Anh - 20215307
"""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
import warnings

warnings.filterwarnings('ignore')

class LanguageClassifier:
    def __init__(self, model_path, max_length=512, device=None):
        """
        Khởi tạo classifier
        Args:
            model_path: Đường dẫn đến model folder
            max_length: Độ dài sequence tối đa
            device: 'cuda' hoặc 'cpu', None = auto detect
        """
        self.model_path = Path(model_path)
        self.max_length = max_length
        
        # Auto detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🖥️  Device: {self.device}")
        
        # Load config
        config_file = self.model_path / 'config.json'
        if not config_file.exists():
            raise FileNotFoundError(f"Không tìm thấy {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
            self.id2label = {int(k): v for k, v in config['id2label'].items()}
            self.label2id = {v: int(k) for k, v in self.id2label.items()}
        
        # Language names
        self.language_names = {
            'vn': 'Vietnamese (Tiếng Việt)',
            'jp': 'Japanese (日本語)',
            'kr': 'Korean (한국어)',
            'us': 'English'
        }
        
        # Load tokenizer và model
        print(f"📥 Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Supported languages: {list(self.id2label.values())}")
    
    def extract_text_from_pdf(self, pdf_path, max_chars=5000):
        """
        Trích xuất text từ PDF
        Args:
            pdf_path: Đường dẫn PDF
            max_chars: Giới hạn ký tự
        Returns:
            str: Text hoặc None nếu lỗi
        """
        try:
            text = extract_text(str(pdf_path))
            
            if not text:
                return None
            
            # Clean
            text = text.strip()
            text = ' '.join(text.split())
            
            # Limit length
            if len(text) > max_chars:
                text = text[:max_chars]
            
            return text
            
        except PDFSyntaxError:
            raise Exception("PDF file bị lỗi hoặc corrupt")
        except Exception as e:
            raise Exception(f"Lỗi khi đọc PDF: {str(e)}")
    
    def predict_from_text(self, text, return_all_scores=True):
        """
        Dự đoán ngôn ngữ từ text
        Args:
            text: Text cần phân loại
            return_all_scores: Trả về xác suất tất cả classes
        Returns:
            dict: Kết quả dự đoán
        """
        # Validate input
        if not text or len(text.strip()) < 10:
            return {
                'success': False,
                'error': 'Text quá ngắn (< 10 ký tự)',
                'language': None,
                'language_name': None,
                'confidence': 0.0
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get prediction
            pred_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][pred_id].item()
        
        # Get predicted language
        predicted_lang = self.id2label[pred_id]
        
        # Result
        result = {
            'success': True,
            'language': predicted_lang,
            'language_name': self.language_names[predicted_lang],
            'confidence': float(confidence),
            'confidence_percent': f"{confidence*100:.2f}%"
        }
        
        # All probabilities
        if return_all_scores:
            all_probs = {}
            for i in range(len(self.id2label)):
                lang = self.id2label[i]
                prob = probabilities[0][i].item()
                all_probs[lang] = {
                    'language_name': self.language_names[lang],
                    'probability': float(prob),
                    'percentage': f"{prob*100:.2f}%"
                }
            result['all_probabilities'] = all_probs
        
        return result
    
    def predict_from_pdf(self, pdf_path, return_text_preview=True):
        """
        Dự đoán ngôn ngữ từ PDF
        Args:
            pdf_path: Đường dẫn PDF
            return_text_preview: Trả về text preview
        Returns:
            dict: Kết quả dự đoán
        """
        # Extract text
        try:
            text = self.extract_text_from_pdf(pdf_path)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': None,
                'confidence': 0.0
            }
        
        if text is None:
            return {
                'success': False,
                'error': 'Không thể trích xuất text từ PDF',
                'language': None,
                'confidence': 0.0
            }
        
        # Predict
        result = self.predict_from_text(text)
        
        # Add text preview
        if return_text_preview and result['success']:
            preview_length = 200
            result['text_preview'] = (
                text[:preview_length] + "..." 
                if len(text) > preview_length 
                else text
            )
            result['text_length'] = len(text)
        
        # Add filename
        result['filename'] = Path(pdf_path).name
        
        return result
    
    def batch_predict(self, pdf_paths, show_progress=True):
        """
        Dự đoán batch nhiều PDFs
        Args:
            pdf_paths: List đường dẫn PDFs
            show_progress: Hiện progress bar
        Returns:
            list: List kết quả
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            pdf_paths = tqdm(pdf_paths, desc="Processing PDFs")
        
        for pdf_path in pdf_paths:
            result = self.predict_from_pdf(pdf_path, return_text_preview=False)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        summary = {
            'total': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': f"{successful/len(results)*100:.1f}%",
            'results': results
        }
        
        return summary


def test_classifier():
    """Test function"""
    # Tìm model mới nhất
    models_dir = Path("models")
    model_folders = [f for f in models_dir.iterdir() 
                     if f.is_dir() and f.name.startswith("xlm-roberta-lang")]
    
    if not model_folders:
        print("❌ Không tìm thấy model! Vui lòng train model trước.")
        return
    
    # Get latest model
    latest_model = sorted(model_folders, key=lambda x: x.name)[-1]
    print(f"📂 Using model: {latest_model.name}\n")
    
    # Load classifier
    classifier = LanguageClassifier(str(latest_model))
    
    # Test với text samples
    print("\n" + "="*70)
    print("🧪 TESTING VỚI TEXT SAMPLES")
    print("="*70)
    
    test_samples = {
        'vn': "Đây là một văn bản tiếng Việt. Chúng ta đang test model phân loại ngôn ngữ.",
        'jp': "これは日本語のテキストです。言語分類モデルをテストしています。",
        'kr': "이것은 한국어 텍스트입니다. 언어 분류 모델을 테스트하고 있습니다.",
        'us': "This is an English text. We are testing the language classification model."
    }
    
    for true_lang, text in test_samples.items():
        result = classifier.predict_from_text(text)
        
        pred_lang = result['language']
        confidence = result['confidence']
        
        status = "✅" if pred_lang == true_lang else "❌"
        
        print(f"\n{status} True: {true_lang.upper()} | Predicted: {pred_lang.upper()} ({confidence*100:.1f}%)")
        print(f"   Text: {text[:60]}...")
    
    # Test với PDF nếu có
    print("\n" + "="*70)
    print("🧪 TESTING VỚI PDF FILES (nếu có)")
    print("="*70)
    
    test_pdfs = list(Path("data/raw").rglob("*.pdf"))[:4]  # Lấy 4 PDFs đầu
    
    if test_pdfs:
        for pdf_path in test_pdfs:
            result = classifier.predict_from_pdf(pdf_path)
            
            if result['success']:
                print(f"\n✅ {result['filename']}")
                print(f"   Language: {result['language_name']}")
                print(f"   Confidence: {result['confidence_percent']}")
            else:
                print(f"\n❌ {result['filename']}")
                print(f"   Error: {result['error']}")
    else:
        print("Không tìm thấy PDF files để test")


if __name__ == "__main__":
    test_classifier()