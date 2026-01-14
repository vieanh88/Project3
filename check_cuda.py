import torch
import sys

print("="*60)
print("🔍 KIỂM TRA CẤU HÌNH GPU")
print("="*60)

print(f"\n📌 Python version: {sys.version}")
print(f"📌 PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"\n✅ CUDA available: TRUE")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   Current GPU: {torch.cuda.current_device()}")
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   Total VRAM: {total_memory:.2f} GB")
    
    # Test tensor
    x = torch.rand(3, 3).cuda()
    print(f"\n✅ Test tensor on GPU: SUCCESS")
    print(f"   Tensor device: {x.device}")
else:
    print("\n❌ CUDA NOT available!")
    print("   Model sẽ chạy trên CPU (rất chậm)")
    print("   Kiểm tra lại:")
    print("   1. NVIDIA driver đã cài đúng chưa?")
    print("   2. PyTorch version có hỗ trợ CUDA không?")

print("\n" + "="*60)