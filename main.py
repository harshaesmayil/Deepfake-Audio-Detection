
"""Telltale: upload a voice clip, get a model's read on whether it is human or synthetic.

Run: python -m uvicorn main:app --host 127.0.0.1 --port 8000
"""

import io
import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample_poly

log = logging.getLogger("telltale")

MODEL_ID = os.getenv(
    "MODEL_ID",
    "Hemgg/Deepfake-audio-detection",
)
MODEL_REVISION = os.getenv(
    "MODEL_REVISION",
    "0d75271368ef2c7efd14831dc503c431f6aab0eb",
)

SAMPLE_RATE = 16000
MIN_SECONDS = float(os.getenv("MIN_SECONDS", "2.0"))
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "15"))
SEGMENT_SECONDS = float(os.getenv("SEGMENT_SECONDS", "13"))
MAX_UPLOAD_BYTES = int(
    os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024))
)

# Decision thresholds:
# AI score >= 0.51: synthetic
# AI score < 0.40: human
# Otherwise: inconclusive
AI_THRESHOLD = float(os.getenv("AI_THRESHOLD", "0.51"))
HUMAN_THRESHOLD = float(os.getenv("HUMAN_THRESHOLD", "0.40"))

# Temperature affects score sharpness, not detection accuracy.
SCORE_TEMPERATURE = float(os.getenv("SCORE_TEMPERATURE", "0.6"))

STATIC_DIR = Path(__file__).parent / "static"


class Detector:
    """Loads the model once and returns adjusted and raw scores per segment."""

    def __init__(self):
        self.ready = False
        self.error = None
        self.extractor = None
        self.model = None
        self.ai_idx = None
        self.human_idx = None

    def load(self):
        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )

            import torch  # noqa: F401

            if SCORE_TEMPERATURE <= 0:
                raise RuntimeError(
                    "SCORE_TEMPERATURE must be greater than 0."
                )

            if not 0 <= HUMAN_THRESHOLD < AI_THRESHOLD <= 1:
                raise RuntimeError(
                    "Thresholds must satisfy "
                    "0 <= HUMAN_THRESHOLD < AI_THRESHOLD <= 1."
                )

            self.extractor = AutoFeatureExtractor.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
            )
            self.model.eval()

            sr = getattr(self.extractor, "sampling_rate", SAMPLE_RATE)
            if sr != SAMPLE_RATE:
                raise RuntimeError(
                    f"Model expects {sr} Hz audio, but the app uses "
                    f"{SAMPLE_RATE} Hz."
                )

            self.ai_idx, self.human_idx = self._resolve_labels(
                self.model.config.id2label
            )

            self.ready = True
            self.error = None

            log.info(
                "Model ready (AI index=%s, human index=%s, temperature=%s, "
                "AI threshold=%s, human threshold=%s)",
                self.ai_idx,
                self.human_idx,
                SCORE_TEMPERATURE,
                AI_THRESHOLD,
                HUMAN_THRESHOLD,
            )

        except Exception as exc:
            self.ready = False
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("Model failed to load")

    @staticmethod
    def _resolve_labels(id2label):
        ai_idx = None
        human_idx = None

        for idx, name in id2label.items():
            label = str(name).lower().replace("_", " ").replace("-", " ")

            if (
                label.startswith("ai")
                or any(
                    keyword in label
                    for keyword in (
                        "fake",
                        "spoof",
                        "synthetic",
                        "generated",
                        "artificial",
                    )
                )
            ):
                ai_idx = int(idx)

            elif any(
                keyword in label
                for keyword in (
                    "human",
                    "real",
                    "bonafide",
                    "bona fide",
                    "genuine",
                    "authentic",
                )
            ):
                human_idx = int(idx)

        if ai_idx is None or human_idx is None:
            raise RuntimeError(
                f"Could not identify the AI and human classes "
                f"from model labels: {id2label}"
            )

        if ai_idx == human_idx:
            raise RuntimeError("AI and human class indices are identical.")

        return ai_idx, human_idx

    def predict(self, segments):
        import torch

        results = []

        for segment in segments:
            inputs = self.extractor(
                segment,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits = self.model(**inputs).logits[0]

            # Original model probabilities.
            raw_probs = torch.softmax(logits, dim=-1)
            raw_ai = float(raw_probs[self.ai_idx])
            raw_human = float(raw_probs[self.human_idx])

            # Temperature-adjusted scores.
            adjusted_probs = torch.softmax(
                logits / SCORE_TEMPERATURE,
                dim=-1,
            )

            adjusted_ai = float(adjusted_probs[self.ai_idx])
            adjusted_human = float(adjusted_probs[self.human_idx])

            results.append(
                {
                    "ai_score": adjusted_ai,
                    "human_score": adjusted_human,
                    "raw_ai_score": raw_ai,
                    "raw_human_score": raw_human,
                }
            )

        return results


detector = Detector()


@asynccontextmanager
async def lifespan(_app):
    if os.getenv("LOAD_MODEL_ON_STARTUP", "true").lower() == "true":
        detector.load()
    yield


app = FastAPI(title="Telltale", lifespan=lifespan)


def to_mono_16k(data: bytes):
    try:
        audio, sample_rate = sf.read(
            io.BytesIO(data),
            dtype="float32",
            always_2d=True,
        )
    except Exception:
        raise HTTPException(
            400,
            "Could not read this audio. Use the web page, which converts "
            "files automatically, or send WAV/FLAC/OGG.",
        )

    mono = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        mono = resample_poly(
            mono,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)

    return mono


def split_segments(mono):
    segment_length = SEGMENT_SECONDS * SAMPLE_RATE
    number_of_segments = max(1, math.ceil(len(mono) / segment_length))
    return np.array_split(mono, number_of_segments)


def decide(scores):
    """Return a stable verdict code and a descriptive display label.

    AI score >= AI_THRESHOLD: synthetic.
    AI score < HUMAN_THRESHOLD: human.
    All other scores: inconclusive.

    The descriptive label indicates score strength, not calibrated certainty.
    """
    mean_ai = float(np.mean([score["ai_score"] for score in scores]))

    if not math.isfinite(mean_ai):
        raise ValueError("The model produced a non-finite AI score.")

    mean_ai = min(1.0, max(0.0, mean_ai))

    if mean_ai >= AI_THRESHOLD:
        verdict = "synthetic"

        if mean_ai >= 0.90:
            label = "Extremely likely synthetic"
        elif mean_ai >= 0.76:
            label = "Very likely synthetic"
        elif mean_ai >= 0.61:
            label = "Highly likely synthetic"
        else:
            label = "Likely synthetic"

    elif mean_ai < HUMAN_THRESHOLD:
        verdict = "human"

        if mean_ai < 0.10:
            label = "Extremely likely human"
        elif mean_ai < 0.20:
            label = "Very likely human"
        elif mean_ai < 0.30:
            label = "Highly likely human"
        else:
            label = "Likely human"

    else:
        verdict = "inconclusive"
        label = "Inconclusive"

    return verdict, label, mean_ai


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "detector_ready": detector.ready,
        "error": detector.error,
    }


@app.get("/api/config")
def config():
    return {
        "min_seconds": MIN_SECONDS,
        "max_seconds": MAX_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "score_temperature": SCORE_TEMPERATURE,
        "ai_threshold": AI_THRESHOLD,
        "human_threshold": HUMAN_THRESHOLD,
    }


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)):
    if not detector.ready:
        raise HTTPException(
            503,
            "The detection model is not loaded yet. "
            + (detector.error or "Give it a moment and retry."),
        )

    data = file.file.read(MAX_UPLOAD_BYTES + 1)

    if not data:
        raise HTTPException(400, "The file is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large.")

    mono = to_mono_16k(data)
    duration = len(mono) / SAMPLE_RATE

    if duration < MIN_SECONDS:
        raise HTTPException(
            400,
            f"Clip is {duration:.1f}s. "
            f"Needs at least {MIN_SECONDS:g}s of audio.",
        )

    truncated = False

    if duration > MAX_SECONDS:
        mono = mono[: int(MAX_SECONDS * SAMPLE_RATE)]
        duration = MAX_SECONDS
        truncated = True

    peak = float(np.max(np.abs(mono)))

    if peak < 1e-4:
        raise HTTPException(400, "The clip is silent.")

    mono = (mono / peak).astype(np.float32)
    segments = split_segments(mono)

    try:
        scores = detector.predict(segments)
        verdict, verdict_label, mean_ai = decide(scores)
    except Exception:
        log.exception("Inference failed")
        raise HTTPException(
            500,
            "Analysis failed while running the model.",
        )

    raw_mean_ai = float(
        np.mean([score["raw_ai_score"] for score in scores])
    )
    raw_mean_human = float(
        np.mean([score["raw_human_score"] for score in scores])
    )

    segment_output = []
    timestamp = 0.0

    for segment, score in zip(segments, scores):
        segment_duration = len(segment) / SAMPLE_RATE

        segment_output.append(
            {
                "start": round(timestamp, 2),
                "end": round(timestamp + segment_duration, 2),
                "ai_score": round(score["ai_score"], 4),
                "human_score": round(score["human_score"], 4),
                "raw_ai_score": round(score["raw_ai_score"], 4),
                "raw_human_score": round(score["raw_human_score"], 4),
            }
        )

        timestamp += segment_duration

    return {
        # Keep this stable for existing frontend code.
        "verdict": verdict,

        # Use this field for the descriptive headline.
        "verdict_label": verdict_label,

        "ai_score": round(mean_ai, 4),
        "human_score": round(1.0 - mean_ai, 4),
        "raw_ai_score": round(raw_mean_ai, 4),
        "raw_human_score": round(raw_mean_human, 4),
        "score_temperature": SCORE_TEMPERATURE,
        "duration": round(duration, 2),
        "truncated": truncated,
        "segments": segment_output,
        "ai_threshold": AI_THRESHOLD,
        "human_threshold": HUMAN_THRESHOLD,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)
