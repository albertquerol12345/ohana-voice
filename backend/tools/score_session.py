import argparse
import json
import os
import re
from pathlib import Path

from backend.config import Config
from backend.matcher import Matcher, normalize_text


FILLER_PREFIXES = re.compile(
    r"^(pongas|pon|ponme|otra|otro|una|un|la|el|quiero|dame|ponmela|ponle)\s+"
)


def load_expected(txt_path: Path, matcher: Matcher, min_score: float, keep_unknown: bool):
    text = txt_path.read_text(encoding="utf-8").strip()
    segments = [seg.strip() for seg in re.split(r"[\\n,]+", text) if seg.strip()]
    expected: list[str] = []
    unknown: list[str] = []
    for seg in segments:
        cleaned = normalize_text(seg)
        cleaned = FILLER_PREFIXES.sub("", cleaned).strip()
        if not cleaned:
            continue
        candidates = matcher.match(
            cleaned, max_tokens=6, ngram_max_tokens=3, topn=1
        )
        if candidates and candidates[0].score >= min_score:
            expected.append(candidates[0].key)
        else:
            if keep_unknown:
                expected.append(f"?{cleaned}")
            else:
                unknown.append(cleaned)
    return expected, unknown


def load_actual(jsonl_path: Path) -> list[str]:
    events: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "event":
                key = obj.get("key")
                if key:
                    events.append(key)
    return events


def lcs_length(expected: list[str], actual: list[str]) -> tuple[int, list[tuple[int, int]]]:
    n, m = len(expected), len(actual)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if expected[i] == actual[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if expected[i - 1] == actual[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return dp[n][m], pairs


def main():
    parser = argparse.ArgumentParser(description="Score a session replay vs transcript")
    parser.add_argument("session", help="Path to session wav (or base name)")
    parser.add_argument("--replay", default=None, help="Replay JSONL path")
    parser.add_argument("--transcript", default=None, help="Transcript TXT path")
    parser.add_argument("--min-score", type=float, default=0.8)
    parser.add_argument("--keep-unknown", action="store_true")
    parser.add_argument("--show", action="store_true", help="Print sequences")
    args = parser.parse_args()

    session_path = Path(args.session)
    if session_path.suffix:
        base = session_path.with_suffix("")
    else:
        base = session_path
    txt_path = Path(args.transcript) if args.transcript else base.with_suffix(".txt")
    replay_path = (
        Path(args.replay)
        if args.replay
        else base.with_name(f"{base.name}_replay.jsonl")
    )

    if not txt_path.exists():
        raise SystemExit(f"No transcript file found: {txt_path}")
    if not replay_path.exists():
        raise SystemExit(f"No replay file found: {replay_path}")

    cfg = Config.from_env()
    keywords_path = cfg.keywords_path or os.path.join(os.getcwd(), "backend", "keywords.json")
    matcher = Matcher(
        keywords_path,
        observed_aliases_path=cfg.observed_aliases_path
        or os.path.join(os.getcwd(), "backend", "keywords_observed.json"),
        include_asr=cfg.include_asr_aliases,
        include_observed=cfg.include_observed_aliases,
    )

    expected, unknown = load_expected(txt_path, matcher, args.min_score, args.keep_unknown)
    actual = load_actual(replay_path)
    lcs_len, pairs = lcs_length(expected, actual)

    precision = lcs_len / len(actual) if actual else 0.0
    recall = lcs_len / len(expected) if expected else 0.0

    print(f"Transcript: {txt_path}")
    print(f"Replay: {replay_path}")
    print(f"Expected: {len(expected)}")
    print(f"Actual: {len(actual)}")
    print(f"LCS: {lcs_len}")
    print(f"Recall: {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Exact match: {expected == actual}")
    if unknown:
        print(f"Unknown segments (skipped): {len(unknown)}")
        if args.show:
            print("Unknown examples:", unknown[:10])

    if args.show:
        print("Expected:", expected)
        print("Actual:", actual)

    if expected and actual and pairs:
        mismatches = []
        exp_i = 0
        act_i = 0
        for ei, ai in pairs:
            while exp_i < ei or act_i < ai:
                mismatches.append((expected[exp_i] if exp_i < ei else None,
                                   actual[act_i] if act_i < ai else None))
                if exp_i < ei:
                    exp_i += 1
                if act_i < ai:
                    act_i += 1
            exp_i = ei + 1
            act_i = ai + 1
        if exp_i < len(expected) or act_i < len(actual):
            mismatches.append((expected[exp_i] if exp_i < len(expected) else None,
                               actual[act_i] if act_i < len(actual) else None))
        if mismatches:
            print("Sample mismatches:", mismatches[:5])


if __name__ == "__main__":
    main()
