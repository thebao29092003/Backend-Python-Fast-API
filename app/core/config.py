from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import Field

# 1. Lấy đường dẫn của thư mục chứa file config.py hiện tại (app/core)
current_dir = Path(__file__).resolve().parent

project_root = current_dir.parent.parent

# 2. Tìm ngược lên thư mục gốc của dự án (đi qua 'core' -> 'app' -> thư mục gốc)
env_path = project_root/ ".env"

class Settings(BaseSettings):
    PROJECT_NAME: str = "Backend Python FastAPI"
    API_V1_STR: str = "/api/v1"

    # Các biến cấu hình lấy từ file .env
    WEB_HOOK_DOT_NET: str
    MODEL_WAV2_VEC2: str

    # Ánh xạ giá trị 'NLTK' từ tệp .env vào biến 'MODEL_NLTK'
    MODEL_NLTK: str = Field(validation_alias="NLTK")

    model_config = SettingsConfigDict(env_file=env_path,
                                      env_file_encoding="utf-8",
                                      extra="ignore")

settings = Settings()