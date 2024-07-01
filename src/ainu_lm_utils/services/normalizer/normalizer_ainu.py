import re
import unicodedata


def strip_accents(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def deduplicate_whitespace(text: str) -> str:
    return " ".join(text.split())


def remove_linking_symbol(text: str) -> str:
    return text.replace("_", "")


def remove_sakehe_symbol(text: str) -> str:
    text = re.sub(r"[VＶⅤ]\d?", "", text)
    text = re.sub(r"\w[VＶⅤ]\w", "", text)
    return text


def remove_aa_ken_annotation(text: str) -> str:
    return re.sub(r"\[\d+\]", "", text)


def remove_biratori_annotation(text: str) -> str:
    return re.sub(r"\[(.*?)\]", "", text)


def remove_koshobungei_annotation(text: str) -> str:
    return text.replace("*", "")


def remove_speaker_annotation(text: str) -> str:
    return re.sub(
        r"^（(川上|萱野|話者|同席者|伊藤|青山|黒川|鍋澤|フチ|貝澤)）", "", text
    )


def remove_glottal_stop_before_affix(text: str) -> str:
    text = re.sub(r"['‘’]=", "=", text)
    text = re.sub(r"=['‘’]", "=", text)
    return text


def remove_ainu_go_archive_annotation(text: str) -> str:
    return text.replace("［注］", "")


def normalize(text: str) -> str:
    text = remove_linking_symbol(text)
    text = remove_sakehe_symbol(text)
    text = remove_aa_ken_annotation(text)
    text = remove_biratori_annotation(text)
    text = remove_koshobungei_annotation(text)
    text = remove_speaker_annotation(text)
    text = remove_glottal_stop_before_affix(text)
    text = remove_ainu_go_archive_annotation(text)

    # These should be the last
    text = strip_accents(text)
    text = deduplicate_whitespace(text)

    return text
