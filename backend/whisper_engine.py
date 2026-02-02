from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from faster_whisper import WhisperModel


@dataclass
class TranscriptionResult:
    text: str
    language: str | None


class WhisperEngine:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        beam_size: int,
        temperature: float,
        hotwords: str | None,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.temperature = temperature
        self.hotwords = hotwords
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio, language: str | None) -> TranscriptionResult:
        segments, info = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            temperature=self.temperature,
            language=language,
            vad_filter=False,
            condition_on_previous_text=False,
            hotwords=self.hotwords or None,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return TranscriptionResult(text=text, language=info.language)


class WhisperManager:
    def __init__(
        self,
        primary_name: str,
        fallback_name: str | None,
        device: str,
        compute_type: str,
        beam_size: int,
        temperature: float,
        hotwords: str | None,
    ):
        self.primary = WhisperEngine(
            primary_name, device, compute_type, beam_size, temperature, hotwords
        )
        self.fallback_name = fallback_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.temperature = temperature
        self.hotwords = hotwords
        self._fallback = None

    def transcribe_primary(self, audio, language: str | None) -> TranscriptionResult:
        return self.primary.transcribe(audio, language)

    def transcribe_fallback(self, audio, language: str | None) -> TranscriptionResult:
        if not self.fallback_name:
            return self.primary.transcribe(audio, language)
        if self._fallback is None:
            self._fallback = WhisperEngine(
                self.fallback_name,
                self.device,
                self.compute_type,
                self.beam_size,
                self.temperature,
                self.hotwords,
            )
        return self._fallback.transcribe(audio, language)
