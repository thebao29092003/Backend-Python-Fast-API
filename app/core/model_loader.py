import torch
from transformers import AutoProcessor, AutoModelForCTC
from app.core.config import settings

class SpeechModel:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        print(f"Khởi động: Đang nạp mô hình Wav2Vec2 vào thiết bị {self.device.upper()}...")
        # Tải Processor và Model từ thư mục local được cấu hình
        self.processor = AutoProcessor.from_pretrained(settings.MODEL_WAV2_VEC2)
        self.model = AutoModelForCTC.from_pretrained(settings.MODEL_WAV2_VEC2).to(self.device)
        print("Mô hình đã sẵn sàng hoạt động!")

    def clear_cache(self):
        # Giải phóng bộ nhớ GPU khi tắt server
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Đã giải phóng bộ nhớ GPU.")

# Khởi tạo đối tượng toàn cục để các tầng khác import và sử dụng
speech_model = SpeechModel()