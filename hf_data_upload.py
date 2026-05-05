from huggingface_hub import HfApi

# Khởi tạo API
api = HfApi()

print("Bắt đầu quá trình upload dữ liệu lên Hugging Face...")

# Sử dụng upload_large_folder thay vì upload_folder
api.upload_large_folder(
    folder_path="D:/Documents/HUST/HUST_Project/Project3/data",        # ĐIỀN VÀO ĐÂY: Đường dẫn đến thư mục chứa data của bạn (vd: "./data")
    repo_id="vieanh/Project3_data_raw",  # ĐIỀN VÀO ĐÂY: Ví dụ "nguyenvana/my-big-dataset"
    repo_type="dataset",                         # Xác định đây là dataset
)

print("Upload thành công! Dữ liệu của bạn đã có trên Hugging Face.")