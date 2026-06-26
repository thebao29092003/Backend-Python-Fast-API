import os
import tempfile
import requests
import librosa
import torch
from app.schemas.convertAudioPhonetic import ConvertAudioPhonetic
from app.core.model_loader import speech_model
from app.core.config import settings

def process_wav2vec2_task(request: ConvertAudioPhonetic):
    path = request.audio_path
    temp_file_path = None

    try:
        # 1. Tải file âm thanh từ URL dưới dạng stream
        response = requests.get(path, stream=True)
        if response.status_code != 200:
            raise Exception("Không thể tải file âm thanh từ URL này!")

        ext = os.path.splitext(path)[1]

        # 2. Tạo file tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            temp_file_path = temp_file.name

        # 3. Đọc và chuẩn hóa tần số lấy mẫu về 16kHz
        speech, sr = librosa.load(temp_file_path, sr=16000)

        # Xóa file tạm ngay sau khi load xong để giải phóng bộ nhớ đĩa
        try:
            os.remove(temp_file_path)
        except Exception as e:
            print(f"Không thể xóa file tạm: {e}")

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

        # 6. Gửi kết quả ngược lại cho hệ thống .NET qua Webhook
        headers = {
            "Content-Type": "application/json",
            "X-Python-Secret": settings.WEB_HOOK_DOT_NET
        }

        callback_payload = {
            "recordingId": request.recording_id,
            "phonemes": phonemes
        }

        print(f"[{request.recording_id}] Đang gửi webhook kết quả về .NET...")
        requests.post(request.callback_url, json=callback_payload, headers=headers, verify=False)

    except Exception as e:
        print(f"[{request.recording_id}] Lỗi trong quá trình xử lý ngầm: {str(e)}")