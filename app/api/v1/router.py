from fastapi import APIRouter
from app.api.v1.endpoints import convert_audio_phonetic_endpoint, text_to_IPA_endpoint

api_router = APIRouter()
api_router.include_router(
    convert_audio_phonetic_endpoint.router,
    prefix="/convert-audio-phonetic", tags=["Convert Audio Phonetic"]
)
api_router.include_router(
    text_to_IPA_endpoint.router,
    prefix="/phonetic-matching", tags=["Phonetic Matching"]
)