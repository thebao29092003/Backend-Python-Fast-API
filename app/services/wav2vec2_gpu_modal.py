import os
import modal

# 1. Định nghĩa Image cho container chạy trên Modal
# Cài đặt ffmpeg qua apt, và các package cần thiết qua pip.
# Đồng thời, dùng run_function để tải trước mô hình và lưu cache vào image giúp khởi động container cực nhanh.
def download_model():
    from transformers import AutoProcessor, AutoModelForCTC
    model_name = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme"
    AutoProcessor.from_pretrained(model_name)
    AutoModelForCTC.from_pretrained(model_name)

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "soundfile",
        "requests",
    )
    .run_function(download_model)
)

# Dòng này đăng ký một ứng dụng trên đám mây của Modal với tên định danh là "wav2vec2-gpu-service".
# Tất cả các container chạy cho app này đều sẽ sử dụng cấu hình image đã định nghĩa ở trên.
app = modal.App("wav2vec2-gpu-service", image=image)

# Sử dụng decorator @app.cls để nạp mô hình một lần duy nhất khi container khởi động trên GPU
@app.cls(gpu="T4", scaledown_window=45)  # GPU T4 có giá thành rẻ, hiệu năng cao và rất tối ưu cho bài toán này
class Wav2Vec2Transcriber:
    # Cơ chế: Hàm load_model được đánh dấu bằng @modal.enter() sẽ chỉ chạy đúng 1 lần duy nhất khi Container vừa được bật lên (Container Start).
    # Nó sẽ nạp mô hình từ đĩa cứng của Container lên VRAM của GPU và lưu vào biến self.model.
    @modal.enter()
    def load_model(self):
        from transformers import AutoProcessor, AutoModelForCTC
        
        self.device = "cuda"
        model_name = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme"
        
        print(f"Đang nạp mô hình Wav2Vec2 lên thiết bị: {self.device.upper()}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name).to(self.device)
        print("Mô hình đã sẵn sàng trên Modal GPU!")

    # Thực thi chuyển audio thành âm vị sử dụng biến self.model đã nạp sẵn trong VRAM
    # từ giai đoạn @modal.enter().
    @modal.method()
    def transcribe(self, audio_url: str) -> str:
        import requests
        import tempfile
        import subprocess
        import librosa
        import torch
        
        temp_audio_path = None
        wav_temp_path = None
        
        try:
            # 1. Tải file âm thanh từ URL
            print(f"Đang tải file âm thanh từ: {audio_url}")
            response = requests.get(audio_url, stream=True, timeout=30)
            if response.status_code != 200:
                raise Exception(f"Không thể tải file âm thanh từ URL! Status code: {response.status_code}")

            # Lấy phần mở rộng đuôi file từ URL
            url_without_params = audio_url.split('?')[0]
            ext = os.path.splitext(url_without_params)[1]
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                temp_audio_path = temp_file.name

            # 2. Chuẩn hóa âm thanh về WAV 16kHz mono bằng FFmpeg
            wav_temp_path = temp_audio_path + ".wav"
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_audio_path,
                "-ar", "16000",
                "-ac", "1",
                "-vn",
                wav_temp_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                stderr_msg = result.stderr.decode('utf-8', errors='ignore')
                raise Exception(f"FFmpeg conversion failed (code {result.returncode}): {stderr_msg}")
            
            # 3. Nạp âm thanh bằng librosa
            speech, sr = librosa.load(wav_temp_path, sr=16000)
            
            # 4. Thực hiện suy luận (Inference) trên GPU
            input_values = self.processor(
                speech, sampling_rate=16000, return_tensors="pt"
            ).input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            phonemes = self.processor.batch_decode(predicted_ids)[0]
            
            print(f"Hoàn thành suy luận. Ký tự phiên âm nhận diện được: {phonemes}")
            return phonemes

        finally:
            # Dọn dẹp tài nguyên file tạm trên Container để tránh rò rỉ dung lượng đĩa
            for path in [temp_audio_path, wav_temp_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Không thể dọn dẹp file tạm {path}: {e}")


@app.local_entrypoint()
def main(audio_url: str):
    transcriber = Wav2Vec2Transcriber()
    print("Đang khởi động thử nghiệm Modal từ xa...")
    phonemes = transcriber.transcribe.remote(audio_url)
    print(f"Kết quả nhận diện thành công!")
    print(f"Phonemes: {phonemes}")