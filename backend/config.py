import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class Config:
    sample_rate: int = 16000

    model_primary: str = "base"
    model_fallback: str = "small"
    enable_fallback: bool = True
    device: str = "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 1
    temperature: float = 0.0

    language_mode: str = "auto"
    language_primary: str | None = None
    language_fallback: str | None = None

    window_sec: float = 1.3
    hop_sec: float = 0.3
    window_queue_max: int = 2
    audio_queue_max: int = 20
    audio_block_sec: float = 0.1
    drop_oldest: bool = True
    record_queue_max: int = 50

    use_gate: bool = True
    gate_factor: float = 1.6
    gate_min_rms: float = 0.006
    gate_hangover_sec: float = 0.7
    gate_noise_alpha: float = 0.05
    gate_noise_floor_init: float = 0.01

    normalize_audio: bool = True
    target_rms: float = 0.10
    max_gain: float = 5.0

    thresh_best: float = 0.82
    thresh_margin: float = 0.08
    margin_relax_score: float = 0.90
    margin_relax: float = 0.04
    margin_strict_score: float = 0.88
    margin_strict: float = 0.10
    margin_override_keys: str = "Big Ohana"
    hysteresis_override_keys: str = "Big Ohana"
    related_key_groups: str = "Ohana|Big Ohana"

    hysteresis_k: int = 2
    hysteresis_window_sec: float = 0.8
    cooldown_sec: float = 0.4
    short_alias_max_len: int = 3
    short_alias_score: float = 0.9
    same_key_block_sec: float = 1.3
    high_confidence_score: float = 0.95
    high_confidence_margin: float = 0.14
    mid_confidence_score: float = 0.85
    mid_confidence_margin: float = 0.16
    mid_confidence_min_len: int = 7

    max_events_per_min: int = 30
    mute_sec: float = 30.0

    max_tokens: int = 3
    ngram_max_tokens: int = 3
    split_on_and: bool = True
    list_mode: bool = True
    list_min_segments: int = 2
    list_cooldown_sec: float = 0.2

    fallback_score_band: float = 0.3
    fallback_margin_low: float = 0.0
    fallback_margin_high: float = 0.2
    max_backlog_for_fallback: int = 1
    fallback_prefer_keys: str = "Big Ohana"
    fallback_prefer_delta: float = 0.05

    debug: bool = False
    debug_topn: int = 5
    state_interval_sec: float = 0.2
    state_send_levels: bool = True
    summary_interval_sec: float = 10.0
    record_session_sec: float = 0.0
    record_session_dir: str = "logs/sessions"

    ws_port: int = 2700
    static_port: int = 8010

    keywords_path: str = ""
    observed_aliases_path: str = ""
    include_asr_aliases: bool = True
    include_observed_aliases: bool = True
    hotwords: str = ""
    hotwords_from_keywords: bool = False

    log_path: str = ""
    profile: str = "balanced"

    audio_device: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()

        cfg.profile = os.environ.get("PROFILE", cfg.profile).lower()
        if cfg.profile == "low_latency":
            cfg.window_sec = 1.1
            cfg.hop_sec = 0.25
            cfg.hysteresis_k = 2
        elif cfg.profile == "robust":
            cfg.window_sec = 1.6
            cfg.hop_sec = 0.4
            cfg.hysteresis_k = 3

        cfg.model_primary = os.environ.get("MODEL_PRIMARY", cfg.model_primary)
        cfg.model_fallback = os.environ.get("MODEL_FALLBACK", cfg.model_fallback)
        cfg.enable_fallback = _env_bool("ENABLE_FALLBACK", cfg.enable_fallback)
        cfg.device = os.environ.get("DEVICE", cfg.device)
        cfg.compute_type = os.environ.get("COMPUTE_TYPE", cfg.compute_type)
        cfg.beam_size = _env_int("BEAM_SIZE", cfg.beam_size)
        cfg.temperature = _env_float("TEMPERATURE", cfg.temperature)

        cfg.language_mode = os.environ.get("LANGUAGE_MODE", cfg.language_mode).lower()
        if cfg.language_mode == "es":
            cfg.language_primary = "es"
        elif cfg.language_mode == "auto":
            cfg.language_primary = None
        elif cfg.language_mode == "dual":
            cfg.language_primary = "es"
            cfg.language_fallback = "en"
        else:
            cfg.language_primary = None

        cfg.window_sec = _env_float("WINDOW_SEC", cfg.window_sec)
        cfg.hop_sec = _env_float("HOP_SEC", cfg.hop_sec)
        cfg.window_queue_max = _env_int("WINDOW_QUEUE_MAX", cfg.window_queue_max)
        cfg.audio_queue_max = _env_int("AUDIO_QUEUE_MAX", cfg.audio_queue_max)
        cfg.audio_block_sec = _env_float("AUDIO_BLOCK_SEC", cfg.audio_block_sec)
        cfg.drop_oldest = _env_bool("DROP_OLDEST", cfg.drop_oldest)
        cfg.record_queue_max = _env_int("RECORD_QUEUE_MAX", cfg.record_queue_max)

        cfg.use_gate = _env_bool("USE_GATE", cfg.use_gate)
        cfg.gate_factor = _env_float("GATE_FACTOR", cfg.gate_factor)
        cfg.gate_min_rms = _env_float("GATE_MIN_RMS", cfg.gate_min_rms)
        cfg.gate_hangover_sec = _env_float("GATE_HANGOVER_SEC", cfg.gate_hangover_sec)
        cfg.gate_noise_alpha = _env_float("GATE_NOISE_ALPHA", cfg.gate_noise_alpha)
        cfg.gate_noise_floor_init = _env_float("GATE_NOISE_FLOOR_INIT", cfg.gate_noise_floor_init)

        cfg.normalize_audio = _env_bool("NORMALIZE_AUDIO", cfg.normalize_audio)
        cfg.target_rms = _env_float("TARGET_RMS", cfg.target_rms)
        cfg.max_gain = _env_float("MAX_GAIN", cfg.max_gain)

        cfg.thresh_best = _env_float("THRESH_BEST", cfg.thresh_best)
        cfg.thresh_margin = _env_float("THRESH_MARGIN", cfg.thresh_margin)
        cfg.margin_relax_score = _env_float("MARGIN_RELAX_SCORE", cfg.margin_relax_score)
        cfg.margin_relax = _env_float("MARGIN_RELAX", cfg.margin_relax)
        cfg.margin_strict_score = _env_float("MARGIN_STRICT_SCORE", cfg.margin_strict_score)
        cfg.margin_strict = _env_float("MARGIN_STRICT", cfg.margin_strict)
        cfg.margin_override_keys = os.environ.get("MARGIN_OVERRIDE_KEYS", cfg.margin_override_keys)
        cfg.hysteresis_override_keys = os.environ.get(
            "HYSTERESIS_OVERRIDE_KEYS", cfg.hysteresis_override_keys
        )
        cfg.related_key_groups = os.environ.get("RELATED_KEY_GROUPS", cfg.related_key_groups)

        cfg.hysteresis_k = _env_int("HYSTERESIS_K", cfg.hysteresis_k)
        cfg.hysteresis_window_sec = _env_float("HYSTERESIS_WINDOW_SEC", cfg.hysteresis_window_sec)
        cfg.cooldown_sec = _env_float("COOLDOWN_SEC", cfg.cooldown_sec)
        cfg.short_alias_max_len = _env_int("SHORT_ALIAS_MAX_LEN", cfg.short_alias_max_len)
        cfg.short_alias_score = _env_float("SHORT_ALIAS_SCORE", cfg.short_alias_score)
        cfg.same_key_block_sec = _env_float("SAME_KEY_BLOCK_SEC", cfg.same_key_block_sec)
        cfg.high_confidence_score = _env_float("HIGH_CONFIDENCE_SCORE", cfg.high_confidence_score)
        cfg.high_confidence_margin = _env_float("HIGH_CONFIDENCE_MARGIN", cfg.high_confidence_margin)
        cfg.mid_confidence_score = _env_float("MID_CONFIDENCE_SCORE", cfg.mid_confidence_score)
        cfg.mid_confidence_margin = _env_float("MID_CONFIDENCE_MARGIN", cfg.mid_confidence_margin)
        cfg.mid_confidence_min_len = _env_int("MID_CONFIDENCE_MIN_LEN", cfg.mid_confidence_min_len)

        cfg.max_events_per_min = _env_int("MAX_EVENTS_PER_MIN", cfg.max_events_per_min)
        cfg.mute_sec = _env_float("MUTE_SEC", cfg.mute_sec)

        cfg.max_tokens = _env_int("MAX_TOKENS", cfg.max_tokens)
        cfg.ngram_max_tokens = _env_int("NGRAM_MAX_TOKENS", cfg.ngram_max_tokens)
        cfg.split_on_and = _env_bool("SPLIT_ON_AND", cfg.split_on_and)
        cfg.list_mode = _env_bool("LIST_MODE", cfg.list_mode)
        cfg.list_min_segments = _env_int("LIST_MIN_SEGMENTS", cfg.list_min_segments)
        cfg.list_cooldown_sec = _env_float("LIST_COOLDOWN_SEC", cfg.list_cooldown_sec)

        cfg.fallback_score_band = _env_float("FALLBACK_SCORE_BAND", cfg.fallback_score_band)
        cfg.fallback_margin_low = _env_float("FALLBACK_MARGIN_LOW", cfg.fallback_margin_low)
        cfg.fallback_margin_high = _env_float("FALLBACK_MARGIN_HIGH", cfg.fallback_margin_high)
        cfg.max_backlog_for_fallback = _env_int(
            "MAX_BACKLOG_FOR_FALLBACK", cfg.max_backlog_for_fallback
        )
        cfg.fallback_prefer_keys = os.environ.get(
            "FALLBACK_PREFER_KEYS", cfg.fallback_prefer_keys
        )
        cfg.fallback_prefer_delta = _env_float(
            "FALLBACK_PREFER_DELTA", cfg.fallback_prefer_delta
        )

        cfg.debug = _env_bool("DEBUG", cfg.debug)
        cfg.debug_topn = _env_int("DEBUG_TOPN", cfg.debug_topn)
        cfg.state_interval_sec = _env_float("STATE_INTERVAL_SEC", cfg.state_interval_sec)
        cfg.state_send_levels = _env_bool("STATE_SEND_LEVELS", cfg.state_send_levels)
        cfg.summary_interval_sec = _env_float("SUMMARY_INTERVAL_SEC", cfg.summary_interval_sec)
        cfg.record_session_sec = _env_float("RECORD_SESSION_SEC", cfg.record_session_sec)
        cfg.record_session_dir = os.environ.get("RECORD_SESSION_DIR", cfg.record_session_dir)

        cfg.ws_port = _env_int("WS_PORT", cfg.ws_port)
        cfg.static_port = _env_int("STATIC_PORT", cfg.static_port)

        cfg.keywords_path = os.environ.get("KEYWORDS_PATH", cfg.keywords_path)
        cfg.observed_aliases_path = os.environ.get("OBSERVED_ALIASES_PATH", cfg.observed_aliases_path)
        cfg.include_asr_aliases = _env_bool("INCLUDE_ASR_ALIASES", cfg.include_asr_aliases)
        cfg.include_observed_aliases = _env_bool(
            "INCLUDE_OBSERVED_ALIASES", cfg.include_observed_aliases
        )
        cfg.hotwords = os.environ.get("HOTWORDS", cfg.hotwords)
        cfg.hotwords_from_keywords = _env_bool("HOTWORDS_FROM_KEYWORDS", cfg.hotwords_from_keywords)

        cfg.log_path = os.environ.get("LOG_PATH", cfg.log_path)
        cfg.audio_device = os.environ.get("AUDIO_DEVICE", cfg.audio_device)

        asr_model = os.path.expanduser("~/asr/models/whisper-large-v3-turbo-es-ct2")
        if "MODEL_FALLBACK" not in os.environ and os.path.exists(asr_model):
            cfg.model_fallback = asr_model
            cfg.enable_fallback = True
            if "BEAM_SIZE" not in os.environ:
                cfg.beam_size = 5
            if "LANGUAGE_MODE" not in os.environ:
                cfg.language_mode = "es"
                cfg.language_primary = "es"
            if "NORMALIZE_AUDIO" not in os.environ:
                cfg.normalize_audio = True
            if "TARGET_RMS" not in os.environ:
                cfg.target_rms = 0.085
            if "MAX_GAIN" not in os.environ:
                cfg.max_gain = 3.5

        return cfg
