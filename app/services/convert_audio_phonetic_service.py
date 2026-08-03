import requests
import modal
from app.schemas.convert_audio_phonetic_schema import ConvertAudioPhoneticRequest
from app.core.config import settings

def process_wav2vec2_task(request: ConvertAudioPhoneticRequest):
    path = request.audio_path

    try:
        # 1. Gọi Modal GPU từ xa để thực hiện xử lý nặng (Tải file, chuẩn hóa và suy luận Wav2Vec2)
        print(f"[{request.recording_id}] Đang gọi Modal GPU từ xa để xử lý âm thanh...")
        env_name = settings.MODAL_ENVIRONMENT

        # Phương thức này tìm kiếm Class trên tài khoản Modal Cloud dựa vào các thông tin:
        # "wav2vec2-gpu-service": Tên của Modal App bạn đã deploy (được khai báo tại app = modal.App("wav2vec2-gpu-service") trong file wav2vec2_gpu_modal.py).
        # "Wav2Vec2Transcriber": Tên Class chứa các hàm xử lý GPU trong Modal App.
        # environment_name=env_name: Môi trường của Modal cần tìm kiếm (ví dụ: development hay production).
        transcriber = modal.Cls.from_name(
            "wav2vec2-gpu-service", 
            "Wav2Vec2Transcriber", 
            environment_name=env_name
        )()

        #  transcriber.transcribe: Gọi hàm transcribe nằm trong class trên.
        # .remote: Chỉ định rằng hàm này sẽ được chạy trên Modal Cloud với thiết lập GPU đã cấu hình.
        phonemes = transcriber.transcribe.remote(audio_url=path)

        # 2. Gửi kết quả ngược lại cho hệ thống .NET qua Webhook
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
        print(f"[{request.recording_id}] Lỗi trong quá trình xử lý trên Modal: {str(e)}")

