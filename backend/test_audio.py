import difflib
import json
import os
import re
import unicodedata
import wave
from pathlib import Path

from vosk import KaldiRecognizer, Model

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = os.environ.get(
    "VOSK_MODEL_PATH", str(ROOT_DIR / "vosk-model-small-es-0.42")
)
KEYWORDS_PATH = Path(__file__).with_name("keywords.json")
MANIFEST_PATH = ROOT_DIR / "audio_samples" / "manifest.json"
USE_GRAMMAR = os.environ.get("VOSK_USE_GRAMMAR", "1") != "0"
INCLUDE_ASR_ALIASES = os.environ.get("VOSK_INCLUDE_ASR_ALIASES", "0") != "0"

SAMPLE_RATE = 16000
MIN_SIMILARITY = 0.6
MIN_MARGIN = 0.0
MIN_PHONETIC_SIMILARITY = 0.6
MAX_FUZZY_TOKENS = 4
STOPWORDS = {
    "a",
    "al",
    "de",
    "del",
    "el",
    "ella",
    "ellas",
    "ellos",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mi",
    "mis",
    "por",
    "favor",
    "porfavor",
    "pon",
    "ponme",
    "dame",
    "quiero",
    "una",
    "un",
    "unas",
    "unos",
    "hamburguesa",
    "burger",
    "burguer",
}
TOKEN_MAP = {
    "by": "bi",
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_match(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    tokens = []
    for token in normalized.split():
        token = TOKEN_MAP.get(token, token)
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return " ".join(tokens)


def phonetic_key(text: str) -> str:
    if not text:
        return ""
    text = normalize_text(text)
    text = text.replace("j", "h")
    text = text.replace("v", "b")
    text = text.replace("z", "s")
    text = text.replace("w", "u")
    text = text.replace("ll", "y")
    text = text.replace("qu", "k")
    text = text.replace("q", "k")
    text = text.replace("c", "k")
    text = text.replace("h", "")
    text = text.replace(" ", "")
    text = re.sub(r"(.)\1+", r"\1", text)
    return text


def load_keywords(use_grammar: bool, include_asr: bool):
    with KEYWORDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    alias_map = {}
    alias_entries = []
    grammar_set = set()

    for key, aliases in data.items():
        if isinstance(aliases, dict):
            phonetic = aliases.get("phonetic") or []
            asr = aliases.get("asr") or []
            alias_list = [(alias, "phonetic") for alias in phonetic]
            if include_asr:
                alias_list.extend((alias, "asr") for alias in asr)
        else:
            alias_list = [(alias, "phonetic") for alias in aliases]

        for alias, source in alias_list:
            alias = alias.strip().lower()
            if not alias:
                continue
            variants = {alias}
            if not alias.startswith("la "):
                variants.add(f"la {alias}")
            for variant in variants:
                normalized = normalize_text(variant)
                if normalized in alias_map:
                    continue
                entry = {
                    "alias": normalized,
                    "key": key,
                    "source": source,
                    "phonetic": phonetic_key(normalized),
                }
                alias_map[normalized] = entry
                alias_entries.append(entry)
                grammar_set.add(variant)

    grammar = None
    if use_grammar:
        grammar_set.add("[unk]")
        grammar = json.dumps(sorted(grammar_set), ensure_ascii=False)
    return alias_map, alias_entries, grammar


def recognize_wav(recognizer: KaldiRecognizer, wav_path: Path):
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate for {wav_path.name}")
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)
        result = json.loads(recognizer.FinalResult())
    return result


def match_keyword(text: str, alias_map: dict, alias_entries: list, use_fuzzy: bool):
    normalized = normalize_text(text)
    cleaned = clean_text_for_match(text)
    if not normalized:
        return None, None

    candidates = []

    def add_candidate(value: str):
        if value and value not in candidates:
            candidates.append(value)

    if cleaned:
        add_candidate(cleaned)
    if normalized and normalized not in candidates:
        add_candidate(normalized)
    if cleaned:
        tokens = cleaned.split()
        if len(tokens) > 1:
            max_len = min(len(tokens), MAX_FUZZY_TOKENS)
            for size in range(1, max_len + 1):
                for index in range(0, len(tokens) - size + 1):
                    add_candidate(" ".join(tokens[index : index + size]))

    for candidate in candidates:
        entry = alias_map.get(candidate)
        if entry:
            return entry, 1.0
    if not use_fuzzy:
        return None, None

    best_key = None
    best_score = 0.0
    second_best = 0.0
    for candidate in candidates:
        if len(candidate.split()) > MAX_FUZZY_TOKENS:
            continue
        for entry in alias_entries:
            score = difflib.SequenceMatcher(None, candidate, entry["alias"]).ratio()
            if score > best_score:
                second_best = best_score
                best_score = score
                best_key = entry["key"]
            elif score > second_best:
                second_best = score

    if best_score >= MIN_SIMILARITY and (best_score - second_best) >= MIN_MARGIN:
        return best_key, best_score
    best_phonetic = None
    best_phonetic_score = 0.0
    for candidate in candidates:
        if len(candidate.split()) > MAX_FUZZY_TOKENS:
            continue
        candidate_phonetic = phonetic_key(candidate)
        if not candidate_phonetic:
            continue
        for entry in alias_entries:
            score = difflib.SequenceMatcher(
                None, candidate_phonetic, entry["phonetic"]
            ).ratio()
            if score > best_phonetic_score:
                best_phonetic_score = score
                best_phonetic = entry["key"]

    if best_phonetic and best_phonetic_score >= MIN_PHONETIC_SIMILARITY:
        return best_phonetic, best_phonetic_score
    return None, None


def main():
    if not Path(MODEL_PATH).is_dir():
        raise SystemExit("Vosk model not found. Set VOSK_MODEL_PATH.")
    if not MANIFEST_PATH.exists():
        raise SystemExit("Missing audio_samples/manifest.json. Run generate_tts.py first.")

    alias_map, alias_entries, grammar = load_keywords(USE_GRAMMAR, INCLUDE_ASR_ALIASES)
    model = Model(MODEL_PATH)

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    passed = 0
    for sample in samples:
        key = sample["key"]
        wav_path = ROOT_DIR / "audio_samples" / sample["file"]
        if USE_GRAMMAR and grammar:
            recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)
        else:
            recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        recognizer.SetWords(True)
        result = recognize_wav(recognizer, wav_path)
        text = (result.get("text") or "").strip()
        entry, similarity = match_keyword(text, alias_map, alias_entries, use_fuzzy=True)
        detected = entry["key"] if entry else None
        ok = detected == key
        status = "OK" if ok else "FAIL"
        sim_text = f"{(similarity or 0):.2f}" if similarity is not None else "-"
        print(f"{status} | expected={key} | heard='{text}' | sim={sim_text} | file={wav_path.name}")
        if ok:
            passed += 1

    total = len(samples)
    print(f"\nAccuracy: {passed}/{total} ({(passed/total*100):.1f}%)")


if __name__ == "__main__":
    main()
