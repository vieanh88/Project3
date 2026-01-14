"""
Training script tối ưu cho RTX 3050 4GB VRAM
Phiên bản tối ưu
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

import wandb # Dùng để theo dõi training (nếu cần)

warnings.filterwarnings('ignore') # Ignore warnings để log gọn hơn

class OptimizedLanguageClassifierTrainer:
    """Trainer được tối ưu cho RTX 3050 4GB VRAM"""
    
    def __init__(self, model_name="xlm-roberta-base", num_labels=4, max_length=512):
        """
        Khởi tạo trainer với cấu hình tối ưu
        
        Args:
            model_name: Tên model từ Hugging Face
            num_labels: Số classes (4: vn/jp/kr/us)
            max_length: Độ dài sequence tối đa (giảm xuống 256 nếu OOM)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._print_device_info()
        self._load_tokenizer()
        self._load_label_mappings()
        
    def _print_device_info(self):
        """In thông tin GPU và clear cache"""
        print("\n" + "="*70)
        print("🖥️  DEVICE INFORMATION")
        print("="*70)
        print(f"Device: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Total VRAM: {total_memory:.2f} GB")
            
            # Clear cache để bắt đầu sạch
            torch.cuda.empty_cache()
            
            # Hiển thị memory trước khi load model
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\nGPU Memory (before loading):")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Available: {total_memory - reserved:.2f} GB")
            print("\n✅ CUDA cache cleared")
        else:
            print("⚠️  WARNING: Running on CPU! Training will be VERY slow.")
            print("   Please ensure CUDA is properly installed.")
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        print(f"\n📥 Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("✅ Tokenizer loaded")
    
    def _load_label_mappings(self):
        """Load label mappings từ file"""
        label_file = 'data/processed/label_mapping.json'
        if not Path(label_file).exists():
            raise FileNotFoundError(
                f"❌ Không tìm thấy {label_file}!\n"
                "   Vui lòng chạy: python src/data_processing.py"
            )
        
        with open(label_file, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
            self.label_map = label_data['label2id']
            self.id2label = {int(k): v for k, v in label_data['id2label'].items()}
            self.language_names = label_data['language_names']
        
        print(f"✅ Label mapping loaded: {list(self.label_map.keys())}")
    
    def load_datasets(self, data_dir='data/processed/splits'):
        """
        Load train/val/test datasets
        
        Returns:
            DatasetDict với train/validation/test
        """
        print("\n" + "="*70)
        print("📂 LOADING DATASETS")
        print("="*70)
        
        data_path = Path(data_dir)
        
        # Check files tồn tại
        required_files = ['train.csv', 'val.csv', 'test.csv']
        for file in required_files:
            if not (data_path / file).exists():
                raise FileNotFoundError(
                    f"❌ Không tìm thấy {data_path / file}!\n"
                    "   Vui lòng chạy: python src/data_processing.py"
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
        """
        Tokenize text
        Không dùng padding ở đây, để DataCollator xử lý (hiệu quả hơn)
        """
        return self.tokenizer(
            examples['text'], 
            truncation=True, 
            max_length=self.max_length,
            padding=False  # Dynamic padding bởi DataCollator
        )
    
    def compute_metrics(self, eval_pred):
        """
        Tính metrics chi tiết cho evaluation
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Overall metrics
        accuracy = accuracy_score(labels, predictions)
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
              # Hyperparameters - TỐI ƯU CHO RTX 3050 4GB
              epochs=3,
              batch_size=6,              # ⚙️ Giảm từ 8 → 6 cho an toàn
              gradient_accumulation=2,   # ⚙️ Effective batch = 6*2 = 12
              learning_rate=2e-5,
              warmup_ratio=0.1,          # ⚙️ 10% warmup
              weight_decay=0.01,
              # Memory optimization
              fp16=True,                 # ⚙️ Mixed precision training
              max_grad_norm=1.0,         # ⚙️ Gradient clipping
              # Evaluation & Saving
              eval_steps=100,            # ⚙️ Evaluate mỗi 100 steps
              save_steps=100,            # ⚙️ Save mỗi 100 steps
              logging_steps=50):         # ⚙️ Log mỗi 50 steps
        """
        Fine-tune model với cấu hình tối ưu cho RTX 3050
        
        Args:
            datasets: DatasetDict
            output_dir: Thư mục lưu model
            epochs: Số epochs
            batch_size: Batch size thực tế (nhỏ hơn để tiết kiệm VRAM)
            gradient_accumulation: Accumulate gradients (tăng effective batch size)
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay
            fp16: Sử dụng mixed precision (tiết kiệm 40-50% VRAM)
            max_grad_norm: Max gradient norm (tránh exploding gradients)
            eval_steps: Evaluate mỗi N steps
            save_steps: Save checkpoint mỗi N steps
            logging_steps: Log mỗi N steps
        """
        print("\n" + "="*70)
        print("🏋️  TRAINING CONFIGURATION")
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
        
        # Load model với torch_dtype tối ưu
        print(f"\n📥 Loading model: {self.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label_map,
            # torch_dtype=torch.float16 if fp16 else torch.float32  # ⚙️ (ValueError: Attempting to unscale FP16 gradients.)
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model loaded")
        print(f"   Total params: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"   Trainable params: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        
        # Data collator với dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Output directory với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"xlm-roberta-lang-{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Tính training schedule
        effective_batch = batch_size * gradient_accumulation
        steps_per_epoch = len(tokenized_datasets['train']) // effective_batch
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        print("\n📊 Training schedule:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size (per device): {batch_size}")
        print(f"   Gradient accumulation: {gradient_accumulation}")
        print(f"   Effective batch size: {effective_batch}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Total steps: {total_steps}")
        print(f"   Warmup steps: {warmup_steps}")
        print(f"   Estimated time: {total_steps * 0.7 / 60:.0f}-{total_steps * 1.2 / 60:.0f} min")
        
        # Training arguments - TỐI ƯU CHO RTX 3050
        training_args = TrainingArguments(
            # Output
            output_dir=str(output_path),
            
            # Evaluation strategy - Đánh giá thường xuyên hơn
            eval_strategy="steps",                    # ⚙️ Eval theo steps thay vì epoch
            eval_steps=eval_steps,                    # ⚙️ Mỗi 100 steps
            
            # Save strategy
            save_strategy="steps",                    # ⚙️ Save theo steps
            save_steps=save_steps,                    # ⚙️ Mỗi 100 steps
            save_total_limit=2,                       # ⚙️ Chỉ giữ 2 checkpoints tốt nhất (tiết kiệm disk)
            load_best_model_at_end=True,              # ⚙️ Load model tốt nhất sau training
            metric_for_best_model="f1",               # ⚙️ Chọn model theo F1 score
            greater_is_better=True,                   # ⚙️ F1 càng cao càng tốt
            
            # Training hyperparameters
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,  # ⚙️ Tăng effective batch size
            
            # Optimization
            learning_rate=learning_rate,
            weight_decay=weight_decay,                # ⚙️ Weight decay để tránh overfitting
            warmup_steps=warmup_steps,                # ⚙️ Warmup để model ổn định
            max_grad_norm=max_grad_norm,              # ⚙️ Clip gradients tránh explode
            optim="adamw_torch",                      # ⚙️ AdamW optimizer của PyTorch
            
            # Performance optimization - QUAN TRỌNG CHO RTX 3050 4GB
            fp16=fp16 and self.device == "cuda",      # ⚙️ Mixed precision (tiết kiệm 40-50% VRAM)
            fp16_opt_level="O1",                      # ⚙️ O1 = conservative mixed precision (ổn định) hoặc O2 = more aggressive (tiết kiệm VRAM hơn nhưng có thể less stable)
            dataloader_num_workers=2,                 # ⚙️ 2 workers cho RTX 3050 (giảm VRAM sử dụng)
            dataloader_pin_memory=True,               # ⚙️ Pin memory tăng tốc độ transfer data lên GPU
            gradient_checkpointing=False,             # ⚙️ Tắt để tăng tốc (trade memory for speed)
            
            # Logging
            logging_dir=str(output_path / 'logs'),
            logging_steps=logging_steps,
            logging_first_step=True,

            # Report to WandB
            report_to="wandb",                                                  # 🔥 BẬT WandB
            run_name=f"xlm-roberta-base-run-bs{batch_size}-lr{learning_rate}",  # 🔥 Tên run trên WandB
            
            # Other settings
            disable_tqdm=False,
            remove_unused_columns=True,
            label_names=["labels"],
            
            # Reproducibility
            seed=42,
            data_seed=42,
        )
        
        print("\n⚙️  Training arguments:")
        print(f"   FP16: {training_args.fp16}")
        print(f"   Max grad norm: {max_grad_norm}")
        print(f"   Gradient checkpointing: {training_args.gradient_checkpointing}")
        print(f"   Dataloader workers: {training_args.dataloader_num_workers}")
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,                # ⚙️ Dừng nếu không cải thiện sau 3 evals
            early_stopping_threshold=0.001            # ⚙️ Threshold để coi là "cải thiện"
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
        
        # Hiển thị GPU memory trước khi train
        if self.device == "cuda":
            print(f"\n💾 GPU Memory before training:")
            print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            free_memory = (torch.cuda.get_device_properties(0).total_memory 
                          - torch.cuda.memory_reserved()) / 1e9
            print(f"   Free: {free_memory:.2f} GB")
        
        # Start training
        print("\n" + "="*70)
        print("🚀 STARTING TRAINING")
        print("="*70)
        print("\n⏱️  Training in progress...\n")
        
        try:
            # Train
            train_result = trainer.train()
            
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETE!")
            print("="*70)
            
            # Hiển thị final metrics
            print("\n📊 Final training metrics:")
            for key, value in train_result.metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
            # Save model và tokenizer
            print(f"\n💾 Saving model to {output_path}...")
            trainer.save_model()
            self.tokenizer.save_pretrained(output_path)
            
            # Save training config
            training_config = {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'batch_size': batch_size,
                'gradient_accumulation': gradient_accumulation,
                'effective_batch_size': effective_batch,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'warmup_ratio': warmup_ratio,
                'weight_decay': weight_decay,
                'fp16': fp16,
                'max_grad_norm': max_grad_norm,
                'train_metrics': train_result.metrics
            }
            
            with open(output_path / 'training_config.json', 'w') as f:
                json.dump(training_config, f, indent=2, default=str)
            
            print("✅ Model and config saved")
            
            # Hiển thị GPU memory sau training
            if self.device == "cuda":
                print(f"\n💾 GPU Memory after training:")
                print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                print(f"   Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            
            return trainer, output_path
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n" + "="*70)
                print("❌ GPU OUT OF MEMORY ERROR!")
                print("="*70)
                print("\n💡 GIẢI PHÁP:")
                print("   1. Giảm batch_size xuống 4 hoặc 3")
                print("   2. Tăng gradient_accumulation lên 3 hoặc 4")
                print("   3. Giảm max_length xuống 256")
                print("   4. Tắt FP16 (fp16=False) - chậm hơn nhưng ít VRAM hơn")
                print("\n📝 Sửa trong hàm train():")
                print("   batch_size=4, gradient_accumulation=3, max_length=256")
                
                # Clear CUDA cache
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    print("\n🔄 CUDA cache cleared")
                
                raise
            else:
                raise
        
        except KeyboardInterrupt:
            print("\n" + "="*70)
            print("⚠️  TRAINING INTERRUPTED BY USER")
            print("="*70)
            print("\n💾 Saving interrupted checkpoint...")
            
            interrupted_path = output_path / "interrupted_checkpoint"
            trainer.save_model(str(interrupted_path))
            self.tokenizer.save_pretrained(interrupted_path)
            
            print(f"✅ Checkpoint saved to: {interrupted_path}")
            print("   You can resume training from this checkpoint later.")
            
            raise
    
    def evaluate_on_test(self, trainer, datasets, output_dir):
        """
        Đánh giá chi tiết trên test set với confusion matrix và per-class metrics
        """
        print("\n" + "="*70)
        print("📊 TEST SET EVALUATION")
        print("="*70)
        
        # Tokenize test set
        print("\n🔄 Tokenizing test set...")
        tokenized_test = datasets['test'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing test"
        )
        
        # Predict
        print("🔮 Predicting on test set...")
        predictions = trainer.predict(tokenized_test)
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = predictions.label_ids
        
        # Overall metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        print("\n" + "="*70)
        print("📈 OVERALL TEST RESULTS")
        print("="*70)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        # Log Confusion Matrix lên WandB
        wandb.log({
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_labels, 
                preds=pred_labels,
                class_names=list(self.language_names.values())
            ),
            "test/accuracy": accuracy,
            "test/f1": f1
        })
        
        # Per-class metrics (chi tiết từng ngôn ngữ)
        print("\n" + "="*70)
        print("📊 PER-LANGUAGE METRICS")
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
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1': float(f1),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
        
        results_file = output_dir / 'test_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Test results saved to: {results_file}")
        
        return results
    
    def _plot_confusion_matrix(self, cm, output_dir):
        """Vẽ confusion matrix đẹp mắt"""
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure với 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        labels = [self.language_names[self.id2label[i]] for i in range(4)]
        
        # Plot 1: Raw counts
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'},
            square=True,
            ax=axes[0]
        )
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Plot 2: Normalized percentages
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Percentage'},
            square=True,
            ax=axes[1]
        )
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        # Save
        cm_path = output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrices saved to: {cm_path}")
        plt.close()


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("🎓 PDF LANGUAGE CLASSIFICATION - OPTIMIZED TRAINING")
    print("="*70)
    
    # ============ CẤU HÌNH - TỐI ƯU CHO RTX 3050 4GB ============
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
    
    print("\n📋 Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    print("\n💡 Note: Nếu gặp Out of Memory:")
    print("   - Giảm batch_size xuống 4 hoặc 3")
    print("   - Tăng gradient_accumulation lên 3 hoặc 4")
    print("   - Giảm max_length xuống 256")
    
    # ============ KHỞI TẠO TRAINER ============
    trainer_obj = OptimizedLanguageClassifierTrainer(
        model_name=CONFIG['model_name'],
        num_labels=4,
        max_length=CONFIG['max_length']
    )

    # Khởi tạo WandB project
    wandb.init(
        project="pdf-language-classification",    # Tên dự án quản lý trên web
        name=f"xlm-roberta-base-run-bs{CONFIG['batch_size']}-lr{CONFIG['learning_rate']}", # Tên lần chạy
        config=CONFIG,                            # Gửi dictionary cấu hình lên để lưu lại
        reinit=True                               # Cho phép chạy lại trong cùng 1 process
    )
    
    # ============ LOAD DATASETS ============
    datasets = trainer_obj.load_datasets()
    
    # ============ TRAINING ============
    trainer, model_path = trainer_obj.train(
        datasets,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation=CONFIG['gradient_accumulation'],
        learning_rate=CONFIG['learning_rate'],
        warmup_ratio=CONFIG['warmup_ratio'],
        weight_decay=CONFIG['weight_decay'],
        fp16=CONFIG['fp16'],
        max_grad_norm=CONFIG['max_grad_norm'],
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps']
    )
    
    # ============ EVALUATION ============
    test_results = trainer_obj.evaluate_on_test(trainer, datasets, model_path)
    
    # ============ HOÀN THÀNH ============
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Saved files:")
    print(f"   Model: {model_path}/")
    print(f"   Config: {model_path}/training_config.json")
    print(f"   Results: {model_path}/test_results.json")
    print(f"   Confusion Matrix: {model_path}/confusion_matrix.png")
    
    print(f"\n📊 Final Test Accuracy: {test_results['test_accuracy']*100:.2f}%")
    print(f"   F1-Score: {test_results['test_f1']:.4f}")
    
    print(f"\n➡️  Next steps:")
    print(f"   1. Review results in: {model_path}/")
    print(f"   2. Test inference: python src/inference.py")
    print(f"   3. Run Streamlit demo: streamlit run app.py")
    
    print("\n" + "="*70)

    # ============ KẾT THÚC WANDB ============
    print("Đang đồng bộ dữ liệu lên WandB...")
    wandb.finish()
    # --------------------------------------


if __name__ == "__main__":
    main()