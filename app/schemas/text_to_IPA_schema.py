from typing import List
from pydantic import BaseModel

# Dữ liệu gửi lên từ Client
class TextToIPARequest(BaseModel):
    word_list: List[str]
    phonemes_list: str

# Cấu trúc điểm số của từng từ trong phản hồi
class WordScore(BaseModel):
    word: str
    correct_phones: int
    total_phones: int
    accuracy: float
    status: str
    original_pronunciation: str
    standard_pronunciation: str

# Dữ liệu phản hồi trả về cho Client
class TextToIPAResponse(BaseModel):
    word_scores: List[WordScore]
    overall_accuracy: float