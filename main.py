import io
import os
import wave
import logging
import secrets
from pathlib import Path
from functools import lru_cache
from typing import Literal

import lameenc
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from piper import PiperVoice

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "en_GB-alan-medium.onnx"

app = FastAPI(
    title="Self-Hosted Piper TTS API",
    description="Text-to-speech API powered by Piper. Output as WAV (default) or MP3.",
    version="1.1.0"
)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    format: Literal["wav", "mp3"] = "wav"


@lru_cache(maxsize=1)
def get_voice():
    if not MODEL_PATH.exists():
        raise RuntimeError("Piper voice model not found.")
    return PiperVoice.load(str(MODEL_PATH))


def verify_api_key(api_key: str | None):
    # .strip() tolerates a stray newline/space in the env var or the header.
    expected_key = (os.getenv("PIPER_API_KEY") or "").strip()

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured."
        )

    if not api_key or not secrets.compare_digest(
        api_key.strip().encode(), expected_key.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )


def wav_to_mp3(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())

    enc = lameenc.Encoder()
    enc.set_bit_rate(64)
    enc.set_in_sample_rate(rate)
    enc.set_channels(channels)
    enc.set_quality(2)
    return bytes(enc.encode(pcm) + enc.flush())


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "Piper TTS API",
        "model_available": MODEL_PATH.exists(),
        "formats": ["wav", "mp3"]
    }


@app.get("/api/voices")
def voices(x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)

    return {
        "voices": [
            {
                "id": "en_GB-alan-medium",
                "language": "English (British)",
                "gender": "Male",
                "quality": "Medium"
            }
        ]
    }


@app.post("/api/tts")
def text_to_speech(
    request: TTSRequest,
    x_api_key: str | None = Header(default=None)
):
    verify_api_key(x_api_key)

    try:
        voice = get_voice()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(request.text, wav_file)
        wav_bytes = buf.getvalue()

        if request.format == "mp3":
            return Response(
                content=wav_to_mp3(wav_bytes),
                media_type="audio/mpeg",
                headers={"Content-Disposition": 'attachment; filename="speech.mp3"'}
            )

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="speech.wav"'}
        )

    except Exception:
        logging.exception("Speech generation failed")
        raise HTTPException(
            status_code=500,
            detail="Speech generation failed."
        )
