# ==========================================
# STAGE 1: Build & Dependency Installation
# ==========================================
FROM python:3.13-slim AS builder

# Tránh ghi file .pyc và cho phép in log trực tiếp ra console
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Cài đặt các công cụ biên dịch cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tạo môi trường ảo riêng biệt
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Sao chép file requirements.txt
COPY requirements.txt .

# Nâng cấp pip và cài đặt PyTorch + Torchaudio phiên bản CPU-only trước để tối ưu dung lượng image
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các gói phụ thuộc còn lại từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Tải trước các tài nguyên NLTK cần thiết (cmudict, averaged_perceptron_tagger) để container chạy offline hoàn toàn
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
    MODEL_WAV2_VEC2="models/wav2vec2"

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho xử lý âm thanh (soundfile/librosa/ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Sao chép môi trường ảo và dữ liệu NLTK từ Stage Builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data

# Tạo thư mục lưu trữ models (sẽ được mount Volume/Bind Mount từ Host vào đây)
RUN mkdir -p /app/models

# Sao chép mã nguồn của dự án (loại trừ các tệp/thư mục trong .dockerignore)
COPY . .

# Tạo user không phải root (security best practice) và phân quyền thư mục /app
RUN useradd -u 1001 -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port ứng dụng
EXPOSE 8000

# Chạy ứng dụng bằng uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
