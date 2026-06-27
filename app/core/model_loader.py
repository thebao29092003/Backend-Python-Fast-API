import torch
from transformers import AutoProcessor, AutoModelForCTC
from app.core.config import settings, project_root
import nltk
from g2p_en import G2p

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

class NLTKModel:
    def __init__(self):
        self.g2p = None

    def load_model(self):
        print("Khởi động: Đang cấu hình NLTK và nạp mô hình G2p...")
        # 1. Chuyển đường dẫn tương đối thành tuyệt đối và chuyển thành dạng chuỗi (string)
        nltk_path = str((project_root / settings.MODEL_NLTK).resolve())

        # 2. Đăng ký trực tiếp đường dẫn này với NLTK
        if nltk_path not in nltk.data.path:
            nltk.data.path.append(nltk_path)

        # 3. Khởi tạo mô hình G2p
        self.g2p = G2p()
        print("Mô hình NLTK G2p đã sẵn sàng!")


# Khởi tạo đối tượng toàn cục để các tầng khác import và sử dụng
speech_model = SpeechModel()
nltk_model = NLTKModel()