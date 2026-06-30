from pydantic import BaseModel

class ConvertAudioPhoneticRequest(BaseModel):
    audio_path: str
    recording_id: str
    transcript_id: str
    callback_url: str