import eng_to_ipa as ipa
import difflib
from app.schemas.text_to_IPA_schema import TextToIPARequest, TextToIPAResponse, WordScore


def calculate_ipa_scores(request: TextToIPARequest) -> TextToIPAResponse:
    words_list = request.word_list
    # Chuyển chuỗi âm vị Wav2Vec2 thành danh sách ký tự
    uttered_phonemes = list(request.phonemes_list)

    # 1. Tạo chuỗi âm vị chuẩn (Target Phonemes) kèm mapping với từ gốc
    target_phonemes = []
    word_mapping = []  # Lưu lại xem âm vị này thuộc từ thứ mấy

    for word_idx, word in enumerate(words_list):
        # Loại bỏ dấu câu trước khi chuyển sang IPA
        clean_word = word.strip(",.?!")
        word_ipa = ipa.convert(clean_word)

        # Loại bỏ các ký tự phụ như dấu trọng âm
        word_ipa = word_ipa.replace("ˈ", "").replace("ˌ", "").replace("*", "")

        for phoneme in word_ipa:
            if phoneme.strip():
                target_phonemes.append(phoneme)
                word_mapping.append(word_idx)

    # 2. Sử dụng SequenceMatcher để so khớp hai chuỗi âm vị
    matcher = difflib.SequenceMatcher(None, target_phonemes, uttered_phonemes)

    # Khởi tạo từ điển lưu trữ kết quả tạm thời của từng từ
    word_scores_map = {
        i: {"word": words_list[i], "total_phones": 0, "correct_phones": 0}
        for i in range(len(words_list))
    }

    # Duyệt qua các khối khớp nhau (matches)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Những âm vị trùng khớp hoàn toàn
            for idx in range(i1, i2):
                w_idx = word_mapping[idx]
                word_scores_map[w_idx]["correct_phones"] += 1
                word_scores_map[w_idx]["total_phones"] += 1
        else:
            # Những âm vị bị phát âm sai, thừa hoặc thiếu
            for idx in range(i1, i2):
                w_idx = word_mapping[idx]
                word_scores_map[w_idx]["total_phones"] += 1

    # 3. Tổng hợp kết quả và đánh giá Đúng/Sai từng từ
    word_scores_list = []
    total_all_correct = 0
    total_all_phones = 0

    for w_idx, data in word_scores_map.items():
        total_phones = data["total_phones"]
        correct_phones = data["correct_phones"]

        # Tránh lỗi chia cho 0 nếu từ đó không phân tích ra được âm vị nào
        accuracy = correct_phones / total_phones if total_phones > 0 else 0.0
        status = "Đúng" if accuracy >= 0.70 else "Sai"

        total_all_correct += correct_phones
        total_all_phones += total_phones

        word_scores_list.append(
            WordScore(
                word=data["word"],
                correct_phones=correct_phones,
                total_phones=total_phones,
                accuracy=round(accuracy, 2),
                status=status
            )
        )

    # Tính độ chính xác tổng thể của cả câu
    overall_accuracy = total_all_correct / total_all_phones if total_all_phones > 0 else 0.0

    return TextToIPAResponse(
        word_scores=word_scores_list,
        overall_accuracy=round(overall_accuracy, 2)
    )