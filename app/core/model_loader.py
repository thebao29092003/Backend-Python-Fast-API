from app.core.config import settings, project_root
import nltk
from g2p_en import G2p

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
nltk_model = NLTKModel()