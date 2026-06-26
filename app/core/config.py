from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# 1. Lấy đường dẫn của thư mục chứa file config.py hiện tại (app/core)
current_dir = Path(__file__).resolve().parent

# 2. Tìm ngược lên thư mục gốc của dự án (đi qua 'core' -> 'app' -> thư mục gốc)
# Cấu trúc: my_speech_project/app/core/config.py
# parent 1: app/core
# parent 2: app/
# parent 3: my_speech_project/ (thư mục gốc chứa .env)
env_path = current_dir.parent.parent / ".env"
class Settings(BaseSettings):
    PROJECT_NAME: str = "Backend pyhon flash API"
    API_V1_STR: str = "/api/v1"

    # Các biến cấu hình lấy từ file .env
    WEB_HOOK_DOT_NET: str
    MODEL_WAV2_VEC2: str

    model_config = SettingsConfigDict(env_file=env_path,
                                      env_file_encoding="utf-8",
                                      extra="ignore")

settings = Settings()