from fastapi import APIRouter, BackgroundTasks
from app.schemas.convert_audio_phonetic_schema import ConvertAudioPhoneticRequest
from app.services.convert_audio_phonetic_service import process_wav2vec2_task

router = APIRouter()

@router.post("/wav2vec2")
async def analyze_audio(request: ConvertAudioPhoneticRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        process_wav2vec2_task,
        request
    )
    return {"statusCode": 202, "message": "processing"}