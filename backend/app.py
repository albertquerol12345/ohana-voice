import asyncio
import os
import queue
import threading
import time
from datetime import datetime, timezone
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from backend.audio_stream import AudioStream, WindowGenerator
from backend.config import Config
from backend.decision import DecisionEngine
from backend.logging_utils import JsonlLogger
from backend.matcher import Matcher
from backend.metrics import Metrics
from backend.processor import Processor
from backend.whisper_engine import WhisperManager
from backend.ws_server import WSBroadcaster

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")


class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)


def start_static_server(port: int):
    httpd = ThreadingHTTPServer(("", port), StaticHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


class SummaryThread(threading.Thread):
    def __init__(self, metrics: Metrics, logger: JsonlLogger, window_queue: queue.Queue, audio: AudioStream, interval: float):
        super().__init__(daemon=True)
        self.metrics = metrics
        self.logger = logger
        self.window_queue = window_queue
        self.audio = audio
        self.interval = interval
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            time.sleep(self.interval)
            snapshot = self.metrics.snapshot()
            snapshot.update(
                {
                    "type": "summary",
                    "backlog": self.window_queue.qsize(),
                    "audio_dropped_frames": self.audio.dropped_frames,
                }
            )
            self.logger.write(snapshot)


async def main():
    config = Config.from_env()
    if not config.keywords_path:
        config.keywords_path = os.path.join(ROOT_DIR, "backend", "keywords.json")
    if not config.observed_aliases_path:
        config.observed_aliases_path = os.path.join(ROOT_DIR, "backend", "keywords_observed.json")
    if not config.log_path:
        config.log_path = os.path.join(ROOT_DIR, "logs", "detections.jsonl")

    logger = JsonlLogger(config.log_path)
    metrics = Metrics()

    matcher = Matcher(
        config.keywords_path,
        observed_aliases_path=config.observed_aliases_path,
        include_asr=config.include_asr_aliases,
        include_observed=config.include_observed_aliases,
    )

    hotwords = config.hotwords
    if config.hotwords_from_keywords and not hotwords:
        try:
            with open(config.keywords_path, "r", encoding="utf-8") as f:
                keys = list(json.load(f).keys())
            hotwords = ", ".join(keys)
        except Exception:
            hotwords = ""

    decision = DecisionEngine(
        thresh_best=config.thresh_best,
        thresh_margin=config.thresh_margin,
        margin_relax_score=config.margin_relax_score,
        margin_relax=config.margin_relax,
        margin_strict_score=config.margin_strict_score,
        margin_strict=config.margin_strict,
        hysteresis_k=config.hysteresis_k,
        hysteresis_window_sec=config.hysteresis_window_sec,
        cooldown_sec=config.cooldown_sec,
        max_events_per_min=config.max_events_per_min,
        mute_sec=config.mute_sec,
        same_key_block_sec=config.same_key_block_sec,
        related_key_groups=config.related_key_groups,
    )

    whisper = WhisperManager(
        primary_name=config.model_primary,
        fallback_name=config.model_fallback if config.enable_fallback else None,
        device=config.device,
        compute_type=config.compute_type,
        beam_size=config.beam_size,
        temperature=config.temperature,
        hotwords=hotwords,
    )

    audio_block_size = max(256, int(config.audio_block_sec * config.sample_rate))
    audio_stream = AudioStream(
        sample_rate=config.sample_rate,
        block_size=audio_block_size,
        queue_max=config.audio_queue_max,
        device=config.audio_device,
        record_queue_max=config.record_queue_max,
    )
    window_queue: queue.Queue = queue.Queue(maxsize=config.window_queue_max)

    loop = asyncio.get_running_loop()
    broadcaster = WSBroadcaster(loop)

    window_generator = WindowGenerator(
        audio_stream=audio_stream,
        window_sec=config.window_sec,
        hop_sec=config.hop_sec,
        ring_sec=config.window_sec + 1.0,
        use_gate=config.use_gate,
        gate_factor=config.gate_factor,
        gate_min_rms=config.gate_min_rms,
        gate_hangover_sec=config.gate_hangover_sec,
        gate_noise_alpha=config.gate_noise_alpha,
        gate_noise_floor_init=config.gate_noise_floor_init,
        normalize_audio=config.normalize_audio,
        target_rms=config.target_rms,
        max_gain=config.max_gain,
        window_queue=window_queue,
        drop_oldest=config.drop_oldest,
        metrics=metrics,
        logger=logger,
        state_cb=broadcaster.publish,
        cooldown_cb=decision.in_cooldown,
        state_interval_sec=config.state_interval_sec,
        state_send_levels=config.state_send_levels,
    )

    processor = Processor(
        window_queue=window_queue,
        whisper=whisper,
        matcher=matcher,
        decision=decision,
        metrics=metrics,
        logger=logger,
        broadcaster=broadcaster,
        config=config,
    )

    summary = SummaryThread(metrics, logger, window_queue, audio_stream, config.summary_interval_sec)

    start_static_server(config.static_port)
    audio_stream.start()
    window_generator.start()
    processor.start()
    summary.start()

    if config.record_session_sec > 0:
        session_id = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")
        audio_path = os.path.join(config.record_session_dir, f"{session_id}.wav")
        if audio_stream.start_recording(audio_path):
            logger.write(
                {
                    "type": "session_start",
                    "session_id": session_id,
                    "audio_path": audio_path,
                    "duration_sec": config.record_session_sec,
                    "ts_mono": time.monotonic(),
                }
            )

            def stop_after():
                time.sleep(config.record_session_sec)
                audio_stream.stop_recording()
                logger.write(
                    {
                        "type": "session_end",
                        "session_id": session_id,
                        "audio_path": audio_path,
                        "duration_sec": config.record_session_sec,
                        "record_dropped_chunks": audio_stream.record_dropped_chunks,
                        "ts_mono": time.monotonic(),
                    }
                )

            threading.Thread(target=stop_after, daemon=True).start()

    print(f"Static server: http://localhost:{config.static_port}")
    print(f"WebSocket server: ws://localhost:{config.ws_port}")
    print(f"Model primary: {config.model_primary} ({config.device}, {config.compute_type})")
    if config.enable_fallback:
        print(f"Model fallback: {config.model_fallback}")

    try:
        await broadcaster.serve("", config.ws_port)
    finally:
        window_generator.stop()
        processor.stop()
        summary.stop()
        audio_stream.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
