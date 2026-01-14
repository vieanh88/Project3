"""
Script xử lý dữ liệu PDF cho dự án phân loại ngôn ngữ
Author: Nguyễn Việt Anh - 20215307
"""

import os # Quản lý hệ thống file
import json # Quản lý file JSON
import warnings # Quản lý cảnh báo
from pathlib import Path # Quản lý đường dẫn
from pdfminer.high_level import extract_text  # pdfminer để trích xuất text
from pdfminer.pdfparser import PDFSyntaxError # Xử lý lỗi PDF
from tqdm import tqdm # Thanh tiến trình
import pandas as pd # Thao tác DataFrame
from sklearn.model_selection import train_test_split # Chia dataset
import matplotlib.pyplot as plt # Vẽ biểu đồ
import seaborn as sns # Thư viện vẽ biểu đồ nâng cao

warnings.filterwarnings('ignore')

class PDFDataProcessor:
    def __init__(self, data_dir, output_dir):
        """
        Khởi tạo processor
        Args:
            data_dir: Thư mục chứa 4 folders (vn/jp/kr/us)
            output_dir: Thư mục lưu kết quả
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping labels
        self.label_map = {
            'vn': 0,  # Tiếng Việt
            'jp': 1,  # Tiếng Nhật
            'kr': 2,  # Tiếng Hàn
            'us': 3   # Tiếng Anh
        }
        
        self.language_names = {
            'vn': 'Vietnamese',
            'jp': 'Japanese',
            'kr': 'Korean',
            'us': 'English'
        }
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'empty_text': 0,
            'errors': []
        }

    # Trích xuất text từ PDF sử dụng pdfminer    
    def extract_text_from_pdf(self, pdf_path, max_chars=5000):
        """
        Trích xuất text từ PDF
        Args:
            pdf_path: Đường dẫn đến file PDF
            max_chars: Giới hạn số ký tự (tránh quá dài)
        Returns:
            str: Text đã trích xuất, hoặc None nếu lỗi
        """
        try:
            # Extract text
            text = extract_text(str(pdf_path))
            
            if not text:
                return None
            
            # Clean text
            text = text.strip()
            text = ' '.join(text.split())  # Normalize whitespace
            
            # Giới hạn độ dài
            if len(text) > max_chars:
                text = text[:max_chars]
            
            return text
            
        except PDFSyntaxError:
            return None
        except Exception as e:
            self.stats['errors'].append({
                'file': pdf_path.name,
                'error': str(e)
            })
            return None
    
    def process_all_pdfs(self, max_samples_per_class=None, min_text_length=100):
        """
        Xử lý tất cả PDFs từ 4 folders
        Args:
            max_samples_per_class: Giới hạn số file mỗi class (None = tất cả)
            min_text_length: Độ dài text tối thiểu
        Returns:
            pd.DataFrame: DataFrame chứa text và labels
        """
        all_data = []
        
        print("\n" + "="*70)
        print("🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU PDF")
        print("="*70)
        
        for folder_name, label_id in self.label_map.items():
            folder_path = self.data_dir / folder_name
            
            if not folder_path.exists():
                print(f"\n⚠️  WARNING: Folder '{folder_path}' không tồn tại!")
                continue
            
            # Lấy tất cả file PDF
            pdf_files = list(folder_path.glob("*.pdf"))
            original_count = len(pdf_files)
            
            # Giới hạn số lượng nếu cần
            if max_samples_per_class and len(pdf_files) > max_samples_per_class:
                pdf_files = pdf_files[:max_samples_per_class]
            
            print(f"\n📁 Xử lý folder: {folder_name.upper()} ({self.language_names[folder_name]})")
            print(f"   Tổng files: {original_count}")
            print(f"   Xử lý: {len(pdf_files)} files")
            
            # Process files với progress bar
            successful = 0
            failed = 0
            empty = 0
            
            for pdf_file in tqdm(pdf_files, desc=f"   {folder_name.upper()}", ncols=70):
                self.stats['total_files'] += 1
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_file)
                
                if text is None:
                    failed += 1
                    self.stats['failed'] += 1
                    continue
                
                if len(text) < min_text_length:
                    empty += 1
                    self.stats['empty_text'] += 1
                    continue
                
                # Add to dataset
                all_data.append({
                    'filename': pdf_file.name,
                    'text': text,
                    'label': label_id,
                    'language': folder_name,
                    'text_length': len(text)
                })
                
                successful += 1
                self.stats['successful'] += 1
            
            # Print summary
            print(f"   ✅ Thành công: {successful}")
            print(f"   ❌ Lỗi: {failed}")
            print(f"   📝 Text quá ngắn: {empty}")
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        print("\n" + "="*70)
        print("📊 TỔNG KẾT XỬ LÝ")
        print("="*70)
        print(f"Tổng files xử lý: {self.stats['total_files']}")
        print(f"Thành công: {self.stats['successful']} ({self.stats['successful']/self.stats['total_files']*100:.1f}%)")
        print(f"Lỗi: {self.stats['failed']}")
        print(f"Text quá ngắn: {self.stats['empty_text']}")
        print(f"\nDataset cuối cùng: {len(df)} samples")
        
        return df
    
    def analyze_dataset(self, df, save_plots=True):
        """
        Phân tích và visualize dataset
        Args:
            df: DataFrame
            save_plots: Có lưu plots không
        """
        print("\n" + "="*70)
        print("📈 PHÂN TÍCH DATASET")
        print("="*70)
        
        # 1. Phân bố số lượng
        print("\n1️⃣  PHÂN BỐ SỐ LƯỢNG:")
        class_counts = df['language'].value_counts().sort_index()
        for lang, count in class_counts.items():
            pct = count / len(df) * 100
            print(f"   {lang.upper()} ({self.language_names[lang]}): {count:,} samples ({pct:.1f}%)")
        
        # 2. Thống kê độ dài text
        print("\n2️⃣  THỐNG KÊ ĐỘ DÀI TEXT:")
        length_stats = df.groupby('language')['text_length'].describe()
        print(length_stats.to_string())
        
        # 3. Ví dụ text
        print("\n3️⃣  VÍ DỤ TEXT TỪ MỖI NGÔN NGỮ:")
        for lang in sorted(self.label_map.keys()):
            if lang in df['language'].values:
                sample = df[df['language'] == lang].iloc[0]
                print(f"\n   [{lang.upper()}] {sample['filename']}")
                preview = sample['text'][:150].replace('\n', ' ')
                print(f"   {preview}...")
        
        # 4. Visualizations
        if save_plots:
            self._create_visualizations(df)
        
        return df
    
    def _create_visualizations(self, df):
        """Tạo các biểu đồ phân tích"""
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # Plot 1: Class distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        class_counts = df['language'].value_counts().sort_index()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        axes[0].bar(range(len(class_counts)), class_counts.values, color=colors)
        axes[0].set_xticks(range(len(class_counts)))
        axes[0].set_xticklabels([f"{lang.upper()}\n({self.language_names[lang]})" 
                                  for lang in class_counts.index])
        axes[0].set_ylabel('Số lượng samples')
        axes[0].set_title('Phân bố số lượng theo ngôn ngữ')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(class_counts.values):
            axes[0].text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')
        
        # Pie chart
        axes[1].pie(class_counts.values, labels=[f"{lang.upper()}\n{v:,}" 
                    for lang, v in zip(class_counts.index, class_counts.values)],
                    colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Tỷ lệ phần trăm theo ngôn ngữ')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Text length distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (lang, color) in enumerate(zip(sorted(self.label_map.keys()), colors)):
            data = df[df['language'] == lang]['text_length']
            ax.hist(data, bins=50, alpha=0.6, label=f'{lang.upper()}', color=color)
        
        ax.set_xlabel('Độ dài text (ký tự)')
        ax.set_ylabel('Số lượng')
        ax.set_title('Phân bố độ dài text theo ngôn ngữ')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'text_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_for_box = [df[df['language'] == lang]['text_length'].values 
                        for lang in sorted(self.label_map.keys())]
        
        bp = ax.boxplot(data_for_box, labels=[lang.upper() for lang in sorted(self.label_map.keys())],
                        patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Độ dài text (ký tự)')
        ax.set_title('Box plot độ dài text theo ngôn ngữ')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'text_length_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Đã lưu plots vào: {plots_dir}")
    
    def create_train_val_test_split(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """
        Chia dataset thành train/val/test với stratified sampling
        Args:
            df: DataFrame
            test_size: Tỷ lệ test set
            val_size: Tỷ lệ validation set
            random_state: Random seed
        Returns:
            train_df, val_df, test_df
        """
        print("\n" + "="*70)
        print("✂️  CHIA DATASET")
        print("="*70)
        
        # Tính tỷ lệ
        total_test_val = test_size + val_size
        
        # Chia train và temp (val+test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=total_test_val,
            stratify=df['label'],
            random_state=random_state
        )
        
        # Chia temp thành val và test
        val_ratio = val_size / total_test_val
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio),
            stratify=temp_df['label'],
            random_state=random_state
        )
        
        # Print summary
        print(f"\nTổng samples: {len(df):,}")
        print(f"\n📊 Phân chia:")
        print(f"   Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"   Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        # Check distribution
        print("\n📈 Phân bố mỗi split:")
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            print(f"\n   {split_name}:")
            counts = split_df['language'].value_counts().sort_index()
            for lang, count in counts.items():
                print(f"      {lang.upper()}: {count:,}")
        
        # Lưu files
        splits_dir = self.output_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        train_df.to_csv(splits_dir / 'train.csv', index=False, encoding='utf-8')
        val_df.to_csv(splits_dir / 'val.csv', index=False, encoding='utf-8')
        test_df.to_csv(splits_dir / 'test.csv', index=False, encoding='utf-8')
        
        print(f"\n✅ Đã lưu splits vào: {splits_dir}/")
        print(f"   - train.csv")
        print(f"   - val.csv")
        print(f"   - test.csv")
        
        return train_df, val_df, test_df
    
    def save_metadata(self):
        """Lưu metadata và label mapping"""
        # Label mapping
        label_file = self.output_dir / 'label_mapping.json'
        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump({
                'label2id': self.label_map,
                'id2label': {str(v): k for k, v in self.label_map.items()},
                'language_names': self.language_names
            }, f, ensure_ascii=False, indent=2)
        
        # Statistics
        stats_file = self.output_dir / 'processing_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Đã lưu metadata:")
        print(f"   - {label_file}")
        print(f"   - {stats_file}")


def main():
    """Main function"""
    # ============ CẤU HÌNH ============
    DATA_DIR = "data/raw"          # Thư mục chứa vn/jp/kr/us
    OUTPUT_DIR = "data/processed"  # Thư mục output
    
    # Khởi tạo processor
    processor = PDFDataProcessor(DATA_DIR, OUTPUT_DIR)
    
    # ============ BƯỚC 1: TRÍCH XUẤT TEXT ============
    print("\n" + "="*70)
    print("BƯỚC 1: TRÍCH XUẤT TEXT TỪ PDF")
    print("="*70)
    
    df = processor.process_all_pdfs(
        max_samples_per_class=10,  # Lấy tất cả, hoặc giới hạn để test nhanh
        min_text_length=100
    )
    
    # Lưu toàn bộ dataset
    full_dataset_path = Path(OUTPUT_DIR) / 'full_dataset.csv'
    df.to_csv(full_dataset_path, index=False, encoding='utf-8')
    print(f"\n✅ Đã lưu full dataset: {full_dataset_path}")
    
    # ============ BƯỚC 2: PHÂN TÍCH ============
    print("\n" + "="*70)
    print("BƯỚC 2: PHÂN TÍCH DỮ LIỆU")
    print("="*70)
    
    df = processor.analyze_dataset(df, save_plots=True)
    
    # ============ BƯỚC 3: CHIA TRAIN/VAL/TEST ============
    print("\n" + "="*70)
    print("BƯỚC 3: CHIA DATASET")
    print("="*70)
    
    train_df, val_df, test_df = processor.create_train_val_test_split(
        df,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # ============ BƯỚC 4: LƯU METADATA ============
    processor.save_metadata()
    
    # ============ HOÀN THÀNH ============
    print("\n" + "="*70)
    print("🎉 HOÀN THÀNH XỬ LÝ DỮ LIỆU!")
    print("="*70)
    print(f"\n📁 Các files đã tạo:")
    print(f"   ✓ {OUTPUT_DIR}/full_dataset.csv")
    print(f"   ✓ {OUTPUT_DIR}/splits/train.csv")
    print(f"   ✓ {OUTPUT_DIR}/splits/val.csv")
    print(f"   ✓ {OUTPUT_DIR}/splits/test.csv")
    print(f"   ✓ {OUTPUT_DIR}/label_mapping.json")
    print(f"   ✓ {OUTPUT_DIR}/processing_stats.json")
    print(f"   ✓ {OUTPUT_DIR}/plots/ (3 biểu đồ)")
    
    print(f"\n➡️  Bước tiếp theo: Chạy training với 'python src/train.py'")


if __name__ == "__main__":
    main()