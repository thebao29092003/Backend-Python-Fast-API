from pydantic import BaseModel

class ConvertAudioPhonetic(BaseModel):
    audio_path: str
    recording_id: str
    callback_url: str