from fastapi import APIRouter
from app.schemas.text_to_IPA_schema import TextToIPAResponse, TextToIPARequest
from app.services.text_to_IPA_service import calculate_ipa_scores

router = APIRouter()

# TỪ TỪ TÍNH SAU
@router.post("/compare", response_model=TextToIPAResponse)
def compare_phonemes(request: TextToIPARequest):
    """
    So khớp chuỗi âm vị Wav2Vec2 thực tế với danh sách từ gốc của văn bản.
    Đánh giá độ chính xác của từng từ dựa trên thuật toán so khớp SequenceMatcher.
    """
    # Gọi hàm xử lý đồng bộ từ tầng Service
    return calculate_ipa_scores(request)

