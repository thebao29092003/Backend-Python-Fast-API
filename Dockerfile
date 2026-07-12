# ==========================================
# STAGE 1: Build & Dependency Installation
# ==========================================
FROM python:3.13-slim AS builder

# Tránh ghi file .pyc và cho phép in log trực tiếp
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# Cài đặt các công cụ biên dịch cần thiết cho một số thư viện Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tạo môi trường ảo riêng biệt để tránh xung đột hệ thống
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Sao chép requirements.txt vào để install dependencies
COPY requirements.txt .

# Nâng cấp pip và cài đặt PyTorch + Torchaudio phiên bản CPU-only trước để tối ưu hóa dung lượng
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện còn lại trong requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Tải trước các tài nguyên NLTK cần thiết (như cmudict) để container có thể chạy offline hoàn toàn
RUN python -m nltk.downloader -d /usr/share/nltk_data cmudict averaged_perceptron_tagger

# ==========================================
# STAGE 2: Lightweight Production Runtime
# ==========================================
FROM python:3.13-slim AS runner

# Thiết lập các biến môi trường runtime tối ưu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    NLTK_DATA="/usr/share/nltk_data" \
    # Ghi đè biến môi trường trong .env để tìm đúng vị trí mô hình Wav2Vec2 từ thư mục gốc
    MODEL_WAV2_VEC2="app/models/wav2vec2"

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho xử lý âm thanh (soundfile/librosa/audioread)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Sao chép môi trường ảo và dữ liệu NLTK từ Stage 1
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data

# Sao chép mã nguồn của dự án (loại trừ các file trong .dockerignore)
COPY . .

# Tạo user không phải root (security best practice) để chạy ứng dụng
RUN useradd -u 1001 -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port ứng dụng
EXPOSE 8000

# Chạy ứng dụng bằng uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
