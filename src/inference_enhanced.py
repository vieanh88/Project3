"""
Enhanced Inference Module với Chunking Strategy
Hỗ trợ trích xuất và xử lý lên đến 50,000 ký tự từ PDF
Author: Nguyễn Việt Anh - 20215307
"""

import torch
import json
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
import warnings

warnings.filterwarnings('ignore')

class EnhancedLanguageClassifier:
    """
    Enhanced classifier với chunking strategy
    Hỗ trợ PDF dài lên đến 50,000 ký tự
    """
    
    def __init__(self, model_path, max_length=512, chunk_size=2000, device=None):
        """
        Khởi tạo enhanced classifier
        
        Args:
            model_path: Đường dẫn model folder
            max_length: Max tokens cho mỗi chunk (512 tokens ~ 2000 chars)
            chunk_size: Số ký tự mỗi chunk (để tránh vượt max_length sau tokenize)
            device: 'cuda' hoặc 'cpu', None = auto detect
        """
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.chunk_size = chunk_size  # Mỗi chunk ~2000 ký tự
        
        # Auto detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🖥️  Device: {self.device}")
        print(f"📏 Chunk size: {self.chunk_size} chars")
        print(f"📏 Max tokens per chunk: {self.max_length}")
        
        # Load config
        self._load_config()
        
        # Load model và tokenizer
        print(f"📥 Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Supported languages: {list(self.id2label.values())}")
    
    def _load_config(self):
        """Load configuration"""
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
    
    def extract_text_from_pdf(self, pdf_path, max_chars=50000):
        """
        Trích xuất text từ PDF - TĂNG LÊN 50,000 KÝ TỰ
        
        Args:
            pdf_path: Đường dẫn PDF
            max_chars: Giới hạn ký tự (50,000)
        Returns:
            str: Text hoặc None nếu lỗi
        """
        try:
            # Extract toàn bộ text (có thể rất dài)
            print(f"📄 Extracting text from PDF...")
            text = extract_text(str(pdf_path))
            
            if not text:
                return None
            
            # Clean text
            text = text.strip()
            text = ' '.join(text.split())  # Normalize whitespace
            
            original_length = len(text)
            
            # Giới hạn độ dài nếu quá dài
            if len(text) > max_chars:
                text = text[:max_chars]
                print(f"⚠️  Text truncated: {original_length:,} → {max_chars:,} chars")
            else:
                print(f"✓ Extracted {len(text):,} chars")
            
            return text
            
        except PDFSyntaxError:
            raise Exception("PDF file bị lỗi hoặc corrupt")
        except Exception as e:
            raise Exception(f"Lỗi khi đọc PDF: {str(e)}")
    
    def split_text_into_chunks(self, text):
        """
        Chia text dài thành nhiều chunks
        
        Strategy:
        - Mỗi chunk ~2000 ký tự (để sau tokenize không vượt 512 tokens)
        - Overlap 200 ký tự giữa các chunks (để không mất context)
        
        Args:
            text: Text dài cần chia
        Returns:
            list: Danh sách chunks
        """
        if len(text) <= self.chunk_size:
            return [text]  # Text ngắn, không cần chia
        
        chunks = []
        overlap = 200  # Overlap 200 chars giữa các chunks
        
        start = 0
        while start < len(text):
            # Lấy chunk
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Nếu không phải chunk cuối, cố gắng cắt ở khoảng trắng
            if end < len(text):
                # Tìm khoảng trắng gần nhất
                last_space = chunk.rfind(' ')
                if last_space > self.chunk_size * 0.8:  # Chỉ cắt nếu không mất quá nhiều text
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk)
            
            # Di chuyển đến chunk tiếp theo (với overlap)
            start = end - overlap
            
            # Tránh loop vô hạn
            if start <= 0 and len(chunks) > 0:
                break
        
        print(f"📝 Split text into {len(chunks)} chunks")
        print(f"   Chunk sizes: {[len(c) for c in chunks[:3]]}{'...' if len(chunks) > 3 else ''}")
        
        return chunks
    
    def predict_single_chunk(self, chunk_text):
        """
        Dự đoán ngôn ngữ cho 1 chunk
        
        Args:
            chunk_text: Text của chunk
        Returns:
            dict: Kết quả dự đoán
        """
        # Tokenize
        inputs = self.tokenizer(
            chunk_text,
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
        
        # All probabilities
        all_probs = {
            self.id2label[i]: probabilities[0][i].item()
            for i in range(len(self.id2label))
        }
        
        return {
            'language': self.id2label[pred_id],
            'confidence': float(confidence),
            'probabilities': all_probs
        }
    
    def aggregate_predictions(self, chunk_results):
        """
        Aggregate kết quả từ nhiều chunks
        
        Strategies:
        1. Majority voting (language xuất hiện nhiều nhất)
        2. Average probabilities (trung bình xác suất)
        3. Weighted by confidence (chunks có confidence cao → weight cao)
        
        Args:
            chunk_results: List kết quả từ các chunks
        Returns:
            dict: Kết quả cuối cùng sau aggregate
        """
        if len(chunk_results) == 1:
            # Chỉ có 1 chunk
            return chunk_results[0]
        
        print(f"\n🔄 Aggregating {len(chunk_results)} chunk predictions...")
        
        # Strategy 1: Majority voting
        languages = [r['language'] for r in chunk_results]
        language_counts = Counter(languages)
        majority_language = language_counts.most_common(1)[0][0]
        
        print(f"   Voting: {dict(language_counts)}")
        
        # Strategy 2: Average probabilities
        avg_probs = {}
        for lang in self.id2label.values():
            probs = [r['probabilities'][lang] for r in chunk_results]
            avg_probs[lang] = np.mean(probs)
        
        # Strategy 3: Weighted average (weight by confidence)
        confidences = [r['confidence'] for r in chunk_results]
        total_confidence = sum(confidences)
        
        weighted_probs = {}
        for lang in self.id2label.values():
            weighted_sum = sum(
                r['probabilities'][lang] * r['confidence']
                for r in chunk_results
            )
            weighted_probs[lang] = weighted_sum / total_confidence
        
        # Final decision: Use weighted probabilities
        final_language = max(weighted_probs, key=weighted_probs.get)
        final_confidence = weighted_probs[final_language]
        
        print(f"   Final: {final_language} (confidence: {final_confidence:.4f})")
        
        return {
            'language': final_language,
            'confidence': float(final_confidence),
            'all_probabilities': weighted_probs,
            'num_chunks': len(chunk_results),
            'majority_vote': majority_language,
            'voting_details': dict(language_counts),
            'chunk_predictions': chunk_results  # Giữ lại để debug
        }
    
    def predict_from_text(self, text, return_all_scores=True, use_chunking=True):
        """
        Dự đoán ngôn ngữ từ text với chunking strategy
        
        Args:
            text: Text cần phân loại
            return_all_scores: Trả về xác suất tất cả classes
            use_chunking: Sử dụng chunking (True) hay predict trực tiếp (False)
        Returns:
            dict: Kết quả dự đoán
        """
        # Validate input
        if not text or len(text.strip()) < 10:
            return {
                'success': False,
                'error': 'Text quá ngắn (< 10 ký tự)',
                'language': None,
                'confidence': 0.0
            }
        
        # Nếu text ngắn hoặc không dùng chunking
        if len(text) <= self.chunk_size or not use_chunking:
            print(f"📊 Processing single chunk ({len(text)} chars)...")
            result = self.predict_single_chunk(text)
            
            return {
                'success': True,
                'language': result['language'],
                'language_name': self.language_names[result['language']],
                'confidence': result['confidence'],
                'confidence_percent': f"{result['confidence']*100:.2f}%",
                'all_probabilities': {
                    lang: {
                        'language_name': self.language_names[lang],
                        'probability': prob,
                        'percentage': f"{prob*100:.2f}%"
                    }
                    for lang, prob in result['probabilities'].items()
                },
                'num_chunks': 1,
                'text_length': len(text)
            }
        
        # Text dài → dùng chunking
        print(f"📊 Processing long text ({len(text):,} chars) with chunking...")
        
        # Chia thành chunks
        chunks = self.split_text_into_chunks(text)
        
        # Predict từng chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            print(f"   Processing chunk {i+1}/{len(chunks)}...", end='\r')
            result = self.predict_single_chunk(chunk)
            chunk_results.append(result)
        
        print()  # New line
        
        # Aggregate kết quả
        aggregated = self.aggregate_predictions(chunk_results)
        
        # Format output
        return {
            'success': True,
            'language': aggregated['language'],
            'language_name': self.language_names[aggregated['language']],
            'confidence': aggregated['confidence'],
            'confidence_percent': f"{aggregated['confidence']*100:.2f}%",
            'all_probabilities': {
                lang: {
                    'language_name': self.language_names[lang],
                    'probability': prob,
                    'percentage': f"{prob*100:.2f}%"
                }
                for lang, prob in aggregated['all_probabilities'].items()
            },
            'num_chunks': aggregated['num_chunks'],
            'majority_vote': aggregated['majority_vote'],
            'voting_details': aggregated['voting_details'],
            'text_length': len(text),
            'chunking_used': True
        }
    
    def predict_from_pdf(self, pdf_path, return_text_preview=True, max_chars=50000):
        """
        Dự đoán ngôn ngữ từ PDF - HỖ TRỢ 50,000 KÝ TỰ
        
        Args:
            pdf_path: Đường dẫn PDF
            return_text_preview: Trả về text preview
            max_chars: Max ký tự trích xuất (50,000)
        Returns:
            dict: Kết quả dự đoán
        """
        print("\n" + "="*70)
        print(f"📄 Processing PDF: {Path(pdf_path).name}")
        print("="*70)
        
        # Extract text với limit 50,000 chars
        try:
            text = self.extract_text_from_pdf(pdf_path, max_chars=max_chars)
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
        
        # Predict với chunking
        result = self.predict_from_text(text, use_chunking=True)
        
        # Add text preview
        if return_text_preview and result['success']:
            preview_length = 5000
            result['text_preview'] = (
                text[:preview_length] + " ..." 
                if len(text) > preview_length 
                else text
            )
        
        # Add filename
        result['filename'] = Path(pdf_path).name
        
        print("\n✅ Processing complete!")
        
        return result


def test_enhanced_classifier():
    """Test function với long text"""
    from pathlib import Path
    
    # Tìm model mới nhất
    models_dir = Path("models")
    model_folders = [f for f in models_dir.iterdir() 
                     if f.is_dir() and f.name.startswith("xlm-roberta-lang")]
    
    if not model_folders:
        print("❌ Không tìm thấy model!")
        return
    
    latest_model = sorted(model_folders, key=lambda x: x.name)[-1]
    print(f"📂 Using model: {latest_model.name}\n")
    
    # Load classifier
    classifier = EnhancedLanguageClassifier(
        str(latest_model),
        chunk_size=2000  # Mỗi chunk 2000 chars
    )
    
    # Test 1: Short text (không chunking)
    print("\n" + "="*70)
    print("TEST 1: SHORT TEXT (No chunking)")
    print("="*70)
    
    short_text = "Đây là văn bản tiếng Việt. " * 20  # ~500 chars
    result = classifier.predict_from_text(short_text)
    print(f"\nResult: {result['language']} ({result['confidence_percent']})")
    print(f"Chunks used: {result.get('num_chunks', 1)}")
    
    # Test 2: Long text (có chunking)
    print("\n" + "="*70)
    print("TEST 2: LONG TEXT (With chunking)")
    print("="*70)
    
    long_text = "Đây là văn bản tiếng Việt rất dài, sử dụng để test với phiên bản xử lý chunk cho pdf dài nhiều ký tự. " * 100  # ~10,000 chars
    result = classifier.predict_from_text(long_text)
    print(f"\nResult: {result['language']} ({result['confidence_percent']})")
    print(f"Chunks used: {result.get('num_chunks', 1)}")
    print(f"Majority vote: {result.get('majority_vote', 'N/A')}")
    print(f"Voting details: {result.get('voting_details', {})}")
    
    # Test 3: PDF file (nếu có)
    print("\n" + "="*70)
    print("TEST 3: PDF FILE (Up to 50,000 chars)")
    print("="*70)
    
    test_pdfs = list(Path("data/raw").rglob("*.pdf"))[:1]
    if test_pdfs:
        result = classifier.predict_from_pdf(test_pdfs[0], max_chars=50000)
        if result['success']:
            print(f"\nResult: {result['language_name']}")
            print(f"Confidence: {result['confidence_percent']}")
            print(f"Text length: {result.get('text_length', 0):,} chars")
            print(f"Chunks processed: {result.get('num_chunks', 1)}")
    else:
        print("No PDF files found for testing")


if __name__ == "__main__":
    test_enhanced_classifier()