from huggingface_hub import HfApi

# Khởi tạo API
api = HfApi()

print("Bắt đầu quá trình upload toàn bộ thư mục models (bao gồm cả checkpoint)...")

api.upload_large_folder(
    # 1. Trỏ đường dẫn tới thư mục gốc mà bạn muốn upload
    folder_path="D:/Documents/HUST/HUST_Project/Project3/models", 
    # 2. Tên Model Repo trên Hugging Face
    repo_id="vieanh/Project3_models",
    # 3. Đảm bảo loại repo là 'model'
    repo_type="model"
    # Bỏ dòng ignore_patterns đi để không loại trừ file nào cả
)

print("Upload thành công! Models của bạn đã có trên Hugging Face.")