import difflib
from app.schemas.text_to_IPA_schema import TextToIPARequest, TextToIPAResponse, WordScore
from app.core.model_loader import nltk_model

# 3. Bảng ánh xạ chuyển đổi Arpabet sang IPA đã đồng bộ hóa hoàn toàn với bộ ký tự của Wav2Vec2
# (Loại bỏ các ký tự độ dài 'ː', đồng bộ hóa turned-r 'ɹ' và r-colored 'ɚ')
ARPABET_TO_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AX': 'ə', 'AY': 'aɪ',
    'EH': 'ɛ', 'ER': 'ɚ', 'EY': 'eɪ', 'IH': 'ɪ', 'IX': 'ɨ', 'IY': 'i', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'ɡ', 'HH': 'h',
    'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p',
    'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v', 'W': 'w',
    'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}


def normalize_ipa(ipa_str: str) -> str:
    """
    Hàm chuẩn hóa để đồng bộ hóa các ký tự IPA giữa bộ G2P chuẩn và mô hình Wav2Vec2 thực tế.
    """
    if not ipa_str:
        return ""

    # Loại bỏ các ký tự dấu phụ, dấu trọng âm, khoảng trắng và tie-bar
    for char in ["ˈ", "ˌ", "*", " ", ",", ".", "?", "!", "͡", "ː"]:
        ipa_str = ipa_str.replace(char, "")

    # Chuyển đổi ký tự chữ g thường (U+0067) sang ký tự g IPA chuẩn (U+0261)
    ipa_str = ipa_str.replace("g", "ɡ")

    # Quy chuẩn các nguyên âm yếu hay bị mô hình Wav2Vec2 gom nhóm để tránh trừ điểm oan
    # (Chuyển schwa 'ə' về 'ʌ' và 'ɨ' về 'ɪ')
    ipa_str = ipa_str.replace("ə", "ʌ")
    ipa_str = ipa_str.replace("ɨ", "ɪ")

    # Chuẩn hóa âm vị R-colored (âm /er/ của giọng Mỹ)
    ipa_str = ipa_str.replace("ɜr", "ɚ").replace("ɜ", "ɚ").replace("ər", "ɚ")

    return ipa_str


def calculate_ipa_scores(request: TextToIPARequest) -> TextToIPAResponse:
    words_list = request.word_list

    g2p_model = nltk_model.g2p

    # --- BƯỚC 1: Xây dựng chuỗi IPA chuẩn cho từng từ từ văn bản gốc ---
    target_words_ipa = []
    for word in words_list:
        clean_word = word.strip(",.?!")
        arpabet_list = g2p_model(clean_word)

        word_ipa_list = []
        for phone in arpabet_list:
            if phone in [' ', ',', '.', '?', '!']:
                continue

            if phone[-1].isdigit():
                phone = phone[:-1]

            ipa_char = ARPABET_TO_IPA.get(phone, "")

            if ipa_char:
                word_ipa_list.append(ipa_char)

        word_ipa_str = "".join(word_ipa_list)
        # Tiến hành chuẩn hóa ký tự cho từ chuẩn
        word_ipa_str = normalize_ipa(word_ipa_str)
        target_words_ipa.append(word_ipa_str)

    # --- BƯỚC 2: Chuẩn hóa chuỗi âm vị thực tế thu được ---
    # Tách chuỗi âm vị thực tế thành các "từ thực tế" dựa trên khoảng trắng
    raw_uttered_tokens = request.phonemes_list.split()
    uttered_tokens = []
    for tok in raw_uttered_tokens:
        cleaned_tok = normalize_ipa(tok.lower())
        if cleaned_tok:
            uttered_tokens.append(cleaned_tok)

    # --- BƯỚC 3: Tiến hành so khớp 2 lớp để tránh lệch pha ---
    # Khởi tạo từ điển lưu trữ điểm của từng từ
    word_scores_map = {
        i: {"word": words_list[i], "total_phones": len(target_words_ipa[i]), "correct_phones": 0}
        for i in range(len(words_list))
    }

    # Lớp 1: So khớp cấp độ Từ (Word-level Alignment)
    word_matcher = difflib.SequenceMatcher(None, target_words_ipa, uttered_tokens)

    for tag, i1, i2, j1, j2 in word_matcher.get_opcodes():
        if tag == 'equal':
            # Trường hợp khớp 1-to-1 giữa từ chuẩn và từ thực tế phát âm
            for idx_t, idx_u in zip(range(i1, i2), range(j1, j2)):
                target_word_ipa = target_words_ipa[idx_t]
                uttered_word_ipa = uttered_tokens[idx_u]

                # Lớp 2: So khớp cấp độ Âm vị (Phoneme-level) cục bộ bên trong từ này
                char_matcher = difflib.SequenceMatcher(None, target_word_ipa, uttered_word_ipa)
                correct = sum(
                    ci2 - ci1 for char_tag, ci1, ci2, cj1, cj2 in char_matcher.get_opcodes() if char_tag == 'equal')

                word_scores_map[idx_t]["correct_phones"] = correct

        elif tag == 'replace':
            # Trường hợp gộp âm nhiều từ (Ví dụ: "in Persian" gộp thành một âm "ɪnɚʃʌn" hoặc phát âm sai)
            target_block_ipa = "".join(target_words_ipa[i1:i2])
            uttered_block_ipa = "".join(uttered_tokens[j1:j2])

            # So khớp cục bộ cho cả khối từ bị gộp
            char_matcher = difflib.SequenceMatcher(None, target_block_ipa, uttered_block_ipa)
            correct = sum(
                ci2 - ci1 for char_tag, ci1, ci2, cj1, cj2 in char_matcher.get_opcodes() if char_tag == 'equal')

            # Phân bổ tỷ lệ âm đúng tương ứng cho từng từ trong khối bị gộp
            total_target_len = len(target_block_ipa)
            match_ratio = correct / total_target_len if total_target_len > 0 else 0.0

            for idx_t in range(i1, i2):
                w_len = word_scores_map[idx_t]["total_phones"]
                word_scores_map[idx_t]["correct_phones"] = int(round(w_len * match_ratio))

        elif tag == 'delete':
            # Người học bỏ sót hoàn toàn từ này, mặc định nhận 0 điểm đúng
            for idx_t in range(i1, i2):
                word_scores_map[idx_t]["correct_phones"] = 0

        elif tag == 'insert':
            # Người nói phát ra âm thừa không có trong văn bản chuẩn, bỏ qua để tránh gây lệch pha
            pass

    # --- BƯỚC 4: Tổng hợp kết quả đầu ra ---
    word_scores_list = []
    total_all_correct = 0
    total_all_phones = 0

    for w_idx, data in word_scores_map.items():
        total_phones = data["total_phones"]
        correct_phones = data["correct_phones"]

        # Nếu từ gốc có tổng số âm vị bằng 0 (ví dụ ký tự đặc biệt), mặc định đạt
        accuracy = correct_phones / total_phones if total_phones > 0 else 1.0
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

    # Tính toán tổng điểm chính xác của toàn đoạn
    overall_accuracy = total_all_correct / total_all_phones if total_all_phones > 0 else 0.0

    return TextToIPAResponse(
        word_scores=word_scores_list,
        overall_accuracy=round(overall_accuracy, 2)
    )