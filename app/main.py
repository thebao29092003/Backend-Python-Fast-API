from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.model_loader import speech_model, nltk_model
from app.api.v1.router import api_router

@asynccontextmanager
async def lifespan(_application: FastAPI):
    # [STARTUP]: Nạp mô hình vào bộ nhớ ngay khi ứng dụng khởi chạy
    speech_model.load_model()
    nltk_model.load_model()

    yield
    # [SHUTDOWN]: Dọn dẹp tài nguyên khi ứng dụng tắt
    speech_model.clear_cache()

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    # khi chạy production ẩn đi
    docs_url="/docs",
)

# Đăng ký các API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Nên dùng chuỗi định dạng "app.main:app" để uvicorn có thể chạy tính năng reload khi dev
    uvicorn.run(app, host="127.0.0.1", port=8000)