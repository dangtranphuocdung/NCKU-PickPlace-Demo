from ultralytics import YOLO

# 1. Load mô hình YOLO bạn đang dùng (Thay đổi tên file .pt nếu dự án của bạn dùng file khác)
# Ví dụ nếu dự án dùng bản mặc định là yolov8n.pt hoặc yolov8s.pt
model = YOLO('yolov8n.pt') 

# 2. KIỂM TRA XEM ĐÃ PRE-TRAINED HAY CHƯA
# Nếu thuộc tính model.ckpt hoặc model.task tồn tại và không trống, nghĩa là trọng số (weights) đã được nạp
if hasattr(model, 'predictor') or model.ckpt:
    print("====== TRẠNG THÁI MÔ HÌNH ======")
    print("👉 Mô hình này ĐÃ ĐƯỢC PRE-TRAINED (Đang có sẵn trọng số weights).")
else:
    print("👉 Mô hình này là mô hình RỖNG (Chưa có weights, chỉ có kiến trúc mạng).")

# 3. KIỂM TRA SỐ LƯỢNG VÀ DANH SÁCH CÁC LABELS (CLASSES)
labels = model.names

print(f"\n====== DANH SÁCH NHÃN (Tổng số: {len(labels)} nhãn) ======")
# In ra định dạng: ID - Tên nhãn tiếng Anh
for class_id, class_name in labels.items():
    print(f"ID {class_id}: {class_name}")

