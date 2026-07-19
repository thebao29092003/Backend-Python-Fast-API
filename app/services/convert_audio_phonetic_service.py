import os
import tempfile
import requests
import librosa
import torch
import subprocess
import threading
from app.schemas.convert_audio_phonetic_schema import ConvertAudioPhoneticRequest
from app.core.model_loader import speech_model
from app.core.config import settings

# Khóa toàn cục để tuần tự hóa việc nạp mô hình & suy luận Wav2Vec2.
# Tránh việc nhiều thread chạy học sâu song song gây nghẽn CPU và tranh chấp GIL.
_wav2vec2_lock = threading.Lock()

def load_audio_robust(file_path: str) -> tuple:
    """
    Nạp file âm thanh một cách tối ưu và mạnh mẽ.
    Sử dụng ffmpeg để chuẩn hóa âm thanh về WAV 16kHz, mono giúp tránh lỗi giải mã mpg123/audioread.
    Nếu không có ffmpeg, tự động fallback về librosa.load thông thường.
    nếu file gốc là audio.mp3, file tạm sẽ là audio.mp3.wav.
    """
    wav_temp_path = file_path + ".wav"
    try:
        # Chuyển đổi file âm thanh sang định dạng wav chuẩn 16kHz, 1 channel (mono) bằng ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", file_path,
            "-ar", "16000",
            "-ac", "1",
            "-vn",
            wav_temp_path
        ]
        
        # Ẩn cửa sổ CMD trên Windows khi gọi subprocess
        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NO_WINDOW
            
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags
        )
        
        if result.returncode == 0 and os.path.exists(wav_temp_path):
            # Vì file đã được chuẩn hóa về 16kHz WAV, librosa.load sẽ nạp trực tiếp qua soundfile cực nhanh
            speech, sr = librosa.load(wav_temp_path, sr=16000)
            return speech, sr
        else:
            stderr_msg = result.stderr.decode('utf-8', errors='ignore')
            print(f"FFmpeg conversion failed (code {result.returncode}). Fallback to librosa.load. Stderr: {stderr_msg}")
    except Exception as e:
        print(f"FFmpeg conversion exception: {e}. Fallback to librosa.load.")
    finally:
        # Xóa file WAV tạm thời sau khi xử lý xong
        if os.path.exists(wav_temp_path):
            try:
                os.remove(wav_temp_path)
            except Exception as ex:
                print(f"Không thể xóa file wav tạm thời: {ex}")

    # Fallback về giải mã mặc định của librosa
    return librosa.load(file_path, sr=16000)

def process_wav2vec2_task(request: ConvertAudioPhoneticRequest):
    path = request.audio_path
    temp_file_path = None

    try:
        # 1. Tải file âm thanh từ URL dưới dạng stream
        response = requests.get(path, stream=True)
        if response.status_code != 200:
            raise Exception("Không thể tải file âm thanh từ URL này!")

        # Loại bỏ các tham số truy vấn trong URL để tách phần mở rộng (extension) chuẩn xác
        url_without_params = path.split('?')[0]
        ext = os.path.splitext(url_without_params)[1]

        # 2. Tạo file tạm thời lưu trữ file tải về
        # nó sẽ tự sinh ra tên tạm cho file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            temp_file_path = temp_file.name

        # 3. Đọc và chuẩn hóa tần số lấy mẫu về 16kHz bằng hàm robust
        speech, sr = load_audio_robust(temp_file_path)
        # Xóa file tạm ngay sau khi load xong để giải phóng bộ nhớ đĩa
        try:
            os.remove(temp_file_path)
            temp_file_path = None
        except Exception as e:
            print(f"Không thể xóa file tạm: {e}")

        # Tuần tự hóa phần xử lý âm thanh và suy luận để giải phóng CPU cho các luồng xử lý khác
        with _wav2vec2_lock:

            # Kiểm tra xem mô hình đã được load thành công qua Lifespan chưa
            if speech_model.processor is None or speech_model.model is None:
                raise Exception("Mô hình chưa được nạp vào hệ thống!")

            # 4. Tiền xử lý dữ liệu và đẩy lên thiết bị (CPU/GPU)
            input_values = speech_model.processor(
                speech, sampling_rate=16000, return_tensors="pt"
            ).input_values.to(speech_model.device)

            # 5. Thực hiện nhận diện (Inference) không tính gradient
            with torch.no_grad():
                logits = speech_model.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            phonemes = speech_model.processor.batch_decode(predicted_ids)[0]

        # 6. Gửi kết quả ngược lại cho hệ thống .NET qua Webhook (nằm ngoài block lock để tăng năng suất)
        headers = {
            "Content-Type": "application/json",
            "X-Python-Secret": settings.WEB_HOOK_DOT_NET
        }

        callback_payload = {
            "recordingId": request.recording_id,
            "transcriptId": request.transcript_id,
            "phonemes": phonemes
        }

        print(f"[{callback_payload}] Đang gửi webhook kết quả về .NET...")
        requests.post(request.callback_url, json=callback_payload, headers=headers, verify=False, timeout=20)

    except Exception as e:
        print(f"[{request.recording_id}] Lỗi trong quá trình xử lý ngầm: {str(e)}")
    finally:
        # Đảm bảo file tạm tải về ban đầu luôn được dọn dẹp sạch sẽ
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Không thể xóa file tạm ở khối finally: {e}")
