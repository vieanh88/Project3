"""
Script training model XLM-RoBERTa cho phân loại ngôn ngữ PDF
Author: Nguyễn Việt Anh - 20215307
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class LanguageClassifierTrainer:
    def __init__(self, model_name="xlm-roberta-base", num_labels=4, max_length=512):
        """
        Khởi tạo trainer
        Args:
            model_name: Tên model từ Hugging Face
            num_labels: Số classes (4: vn/jp/kr/us)
            max_length: Độ dài sequence tối đa
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("\n" + "="*70)
        print("🖥️  KIỂM TRA THIẾT BỊ")
        print("="*70)
        print(f"Device: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"VRAM: {total_memory:.2f} GB")
            
            # Clear cache
            torch.cuda.empty_cache()
            print("✅ CUDA cache cleared")
        else:
            print("⚠️  WARNING: Đang chạy trên CPU! Training sẽ rất chậm.")
        
        # Load tokenizer
        print(f"\n📥 Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded")
        
        # Load label mapping
        label_file = 'data/processed/label_mapping.json'
        if not Path(label_file).exists():
            raise FileNotFoundError(
                f"Không tìm thấy {label_file}! "
                "Vui lòng chạy data_processing.py trước."
            )
        
        with open(label_file, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
            self.label_map = label_data['label2id']
            self.id2label = {int(k): v for k, v in label_data['id2label'].items()}
            self.language_names = label_data['language_names']
        
        print(f"✅ Label mapping loaded: {list(self.label_map.keys())}")
        
    def load_datasets(self, data_dir='data/processed/splits'):
        """Load train/val/test datasets"""
        print("\n" + "="*70)
        print("📂 LOADING DATASETS")
        print("="*70)
        
        data_path = Path(data_dir)
        
        # Check files exist
        required_files = ['train.csv', 'val.csv', 'test.csv']
        for file in required_files:
            if not (data_path / file).exists():
                raise FileNotFoundError(
                    f"Không tìm thấy {data_path / file}! "
                    "Vui lòng chạy data_processing.py trước."
                )
        
        # Load CSVs
        print("Loading CSV files...")
        train_df = pd.read_csv(data_path / 'train.csv')
        val_df = pd.read_csv(data_path / 'val.csv')
        test_df = pd.read_csv(data_path / 'test.csv')
        
        # Convert to HuggingFace Dataset
        datasets = DatasetDict({
            'train': Dataset.from_pandas(train_df[['text', 'label']]),
            'validation': Dataset.from_pandas(val_df[['text', 'label']]),
            'test': Dataset.from_pandas(test_df[['text', 'label']])
        })
        
        print(f"✅ Datasets loaded:")
        print(f"   Train:      {len(datasets['train']):,} samples")
        print(f"   Validation: {len(datasets['validation']):,} samples")
        print(f"   Test:       {len(datasets['test']):,} samples")
        
        return datasets
    
    def preprocess_function(self, examples):
        """Tokenize text"""
        return self.tokenizer(
            examples['text'], 
            truncation=True, 
            max_length=self.max_length,
            padding=False  # Dynamic padding
        )
    
    def compute_metrics(self, eval_pred):
        """Tính metrics cho evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, datasets, output_dir='models', 
              epochs=3, batch_size=8, learning_rate=2e-5,
              warmup_ratio=0.1, weight_decay=0.01):
        """
        Fine-tune model
        Args:
            datasets: DatasetDict với train/val/test
            output_dir: Thư mục lưu model
            epochs: Số epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_ratio: Tỷ lệ warmup steps
            weight_decay: Weight decay cho optimizer
        """
        print("\n" + "="*70)
        print("🏋️  BẮT ĐẦU TRAINING")
        print("="*70)
        
        # Tokenize datasets
        print("\n🔄 Tokenizing datasets...")
        tokenized_datasets = datasets.map(
            self.preprocess_function, 
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing"
        )
        print("✅ Tokenization complete")
        
        # Load model
        print(f"\n📥 Loading model: {self.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label_map
        )
        print("✅ Model loaded")
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"xlm-roberta-lang-{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=True if self.device == "cuda" else False,
            logging_dir=str(output_path / 'logs'),
            logging_steps=50,
            logging_first_step=True,
            save_total_limit=2,
            report_to="none",
            seed=42,
            data_seed=42,
            remove_unused_columns=True,
        )
        
        # Early stopping
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.001
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )
        
        # Print config
        print("\n📋 TRAINING CONFIGURATION:")
        print(f"   Model: {self.model_name}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Max length: {self.max_length}")
        print(f"   Weight decay: {weight_decay}")
        print(f"   Warmup ratio: {warmup_ratio}")
        print(f"   FP16: {training_args.fp16}")
        print(f"   Output: {output_path}")
        
        # Estimate training time
        num_train_samples = len(tokenized_datasets['train'])
        steps_per_epoch = num_train_samples // (batch_size * torch.cuda.device_count() if torch.cuda.is_available() else batch_size)
        total_steps = steps_per_epoch * epochs
        print(f"\n⏱️  Estimated:")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Total steps: {total_steps}")
        print(f"   Training time: ~{total_steps * 0.5 / 60:.0f}-{total_steps * 1 / 60:.0f} minutes")
        
        # Start training
        print("\n🚀 Starting training...\n")
        train_result = trainer.train()
        
        # Save final model
        print("\n💾 Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_path)
        
        # Save training args
        training_args_dict = training_args.to_dict()
        with open(output_path / 'training_config.json', 'w') as f:
            json.dump(training_args_dict, f, indent=2)
        
        print(f"\n✅ Training complete!")
        print(f"   Best model saved to: {output_path}")
        
        return trainer, output_path
    
    def evaluate_on_test(self, trainer, datasets, output_dir):
        """Đánh giá chi tiết trên test set"""
        print("\n" + "="*70)
        print("📊 ĐÁNH GIÁ TRÊN TEST SET")
        print("="*70)
        
        # Tokenize test set
        tokenized_test = datasets['test'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Predict
        print("\n🔮 Predicting...")
        predictions = trainer.predict(tokenized_test)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Overall metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        print("\n" + "="*70)
        print("📈 KẾT QUẢ TỔNG QUAN")
        print("="*70)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Per-class metrics
        print("\n" + "="*70)
        print("📊 METRICS CHI TIẾT THEO NGÔN NGỮ")
        print("="*70)
        
        class_report = classification_report(
            true_labels, pred_labels,
            target_names=[self.language_names[self.id2label[i]] for i in range(4)],
            digits=4
        )
        print(class_report)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        self._plot_confusion_matrix(cm, output_dir)
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
        
        results_file = output_dir / 'test_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        return results
    
    def _plot_confusion_matrix(self, cm, output_dir):
        """Vẽ confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        labels = [self.language_names[self.id2label[i]] for i in range(4)]
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'},
            square=True
        )
        
        plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save
        cm_path = output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {cm_path}")
        plt.close()
        
        # Also create normalized version
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Percentage'},
            square=True
        )
        
        plt.title('Normalized Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        cm_norm_path = output_dir / 'confusion_matrix_normalized.png'
        plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
        print(f"✅ Normalized confusion matrix saved to: {cm_norm_path}")
        plt.close()


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("🎓 PDF LANGUAGE CLASSIFICATION - TRAINING")
    print("="*70)
    
    # ============ CẤU HÌNH ============
    CONFIG = {
        'model_name': 'xlm-roberta-base',
        'max_length': 512, # Giảm xuống 256 nếu hết VRAM
        'epochs': 3,
        'batch_size': 8,  # Giảm xuống 4 nếu hết VRAM
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01
    }
    
    print("\n📋 Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # ============ KHỞI TẠO TRAINER ============
    trainer_obj = LanguageClassifierTrainer(
        model_name=CONFIG['model_name'],
        num_labels=4,
        max_length=CONFIG['max_length']
    )
    
    # ============ LOAD DATASETS ============
    datasets = trainer_obj.load_datasets()
    
    # ============ TRAINING ============
    try:
        trainer, model_path = trainer_obj.train(
            datasets,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            learning_rate=CONFIG['learning_rate'],
            warmup_ratio=CONFIG['warmup_ratio'],
            weight_decay=CONFIG['weight_decay']
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ ERROR: Out of VRAM!")
            print("💡 Giải pháp:")
            print("   1. Giảm batch_size xuống 4 hoặc 2")
            print("   2. Giảm max_length xuống 256")
            print("   3. Tắt các ứng dụng khác đang dùng GPU")
            raise
        else:
            raise
    
    # ============ EVALUATION ============
    test_results = trainer_obj.evaluate_on_test(trainer, datasets, model_path)
    
    # ============ HOÀN THÀNH ============
    print("\n" + "="*70)
    print("🎉 TRAINING HOÀN THÀNH!")
    print("="*70)
    print(f"\n📁 Model & results:")
    print(f"   Model: {model_path}")
    print(f"   Config: {model_path}/training_config.json")
    print(f"   Results: {model_path}/test_results.json")
    print(f"   Confusion Matrix: {model_path}/confusion_matrix.png")
    
    print(f"\n📊 Final Test Accuracy: {test_results['accuracy']*100:.2f}%")
    
    print(f"\n➡️  Bước tiếp theo:")
    print(f"   1. Kiểm tra kết quả trong {model_path}")
    print(f"   2. Chạy Streamlit demo: streamlit run app.py")


if __name__ == "__main__":
    main()