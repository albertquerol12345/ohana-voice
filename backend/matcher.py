import json
import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_alt(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("v", "b").replace("z", "s")
    text = text.replace("w", "u")
    tokens = []
    for token in text.split():
        if token.startswith("h") and len(token) > 2:
            token = token[1:]
        tokens.append(token)
    return " ".join(tokens)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def similarity_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    dist = levenshtein_distance(a, b)
    lev = 1.0 - (dist / max(len(a), len(b)))
    return max(seq, lev)


def generate_ngrams(tokens: list[str], max_len: int) -> list[str]:
    grams = []
    n_tokens = len(tokens)
    for length in range(1, max_len + 1):
        for start in range(0, n_tokens - length + 1):
            grams.append(" ".join(tokens[start : start + length]))
    return grams


@dataclass
class AliasEntry:
    key: str
    alias: str
    norm: str
    norm_alt: str
    tokens: tuple[str, ...]
    token_count: int


@dataclass
class Candidate:
    key: str
    score: float
    alias: str
    ngram: str


class Matcher:
    def __init__(
        self,
        keywords_path: str,
        observed_aliases_path: str | None = None,
        include_asr: bool = True,
        include_observed: bool = True,
    ):
        self.aliases: list[AliasEntry] = []
        self._load_keywords(keywords_path, include_asr)
        if observed_aliases_path and include_observed and os.path.isfile(observed_aliases_path):
            self._load_observed(observed_aliases_path)

    def _add_alias(self, key: str, alias: str):
        alias = alias.strip()
        if not alias:
            return
        norm = normalize_text(alias)
        norm_alt = normalize_text_alt(alias)
        tokens = tuple(norm.split())
        self.aliases.append(
            AliasEntry(
                key=key,
                alias=alias,
                norm=norm,
                norm_alt=norm_alt,
                tokens=tokens,
                token_count=len(tokens),
            )
        )

    def _load_keywords(self, path: str, include_asr: bool):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, payload in data.items():
            phonetic = payload.get("phonetic", [])
            for alias in phonetic:
                self._add_alias(key, alias)
            if include_asr:
                for alias in payload.get("asr", []):
                    self._add_alias(key, alias)

    def _load_observed(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, aliases in data.items():
            for alias in aliases:
                self._add_alias(key, alias)

    def match(self, text: str, max_tokens: int, ngram_max_tokens: int, topn: int = 5) -> list[Candidate]:
        normalized = normalize_text(text)
        tokens = normalized.split()
        if not tokens:
            return []

        ngrams = generate_ngrams(tokens, ngram_max_tokens)
        if len(tokens) <= max_tokens and normalized not in ngrams:
            ngrams.append(normalized)

        best_by_key: dict[str, Candidate] = {}
        for gram in ngrams:
            if not gram:
                continue
            gram_tokens = gram.split()
            gram_token_count = len(gram_tokens)
            gram_token_set = set(gram_tokens)
            for alias in self.aliases:
                score = max(
                    similarity_score(gram, alias.norm),
                    similarity_score(gram, alias.norm_alt),
                )
                if gram_token_count > 1 and alias.token_count == 1:
                    score *= 0.98
                if alias.token_count > 1 and alias.token_count <= gram_token_count:
                    if all(token in gram_token_set for token in alias.tokens):
                        score += 0.02
                if score and len(alias.norm) <= 3 and len(gram) <= 3:
                    score = min(1.0, score + 0.05)
                current = best_by_key.get(alias.key)
                if current is None or score > current.score:
                    best_by_key[alias.key] = Candidate(
                        key=alias.key, score=score, alias=alias.alias, ngram=gram
                    )

        candidates = sorted(best_by_key.values(), key=lambda item: item.score, reverse=True)
        return candidates[:topn]
