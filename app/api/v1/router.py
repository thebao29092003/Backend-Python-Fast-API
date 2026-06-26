from fastapi import APIRouter
from app.api.v1.endpoints import convertAudioPhonetic

api_router = APIRouter()
api_router.include_router(convertAudioPhonetic.router, prefix="/convert-audio-phonetic", tags=["convertAudioPhonetic"])