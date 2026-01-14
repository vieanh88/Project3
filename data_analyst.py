import os

# Đường dẫn tới folder gốc
BASE_DIR = "data/raw"
# Các folder ngôn ngữ
LANG_FOLDERS = ["jp", "kr", "us", "vn"]

def count_pdfs_recursive(folder_path):
    pdf_count = 0
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_count += 1
    return pdf_count

stats = {}

for lang in LANG_FOLDERS:
    lang_path = os.path.join(BASE_DIR, lang)

    if not os.path.exists(lang_path):
        print(f"⚠️ Folder không tồn tại: {lang_path}")
        stats[lang] = 0
        continue

    pdf_count = count_pdfs_recursive(lang_path)
    stats[lang] = pdf_count

print("📊 THỐNG KÊ SỐ LƯỢNG FILE PDF\n")

total = 0
for lang, count in stats.items():
    print(f"{lang.upper():>2} : {count:,} files PDF")
    total += count

print("\n-----------------------------")
print(f"TỔNG : {total:,} files PDF")