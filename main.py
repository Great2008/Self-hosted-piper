import io
import os
import re
import wave
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
    description="Text-to-speech API powered by Piper.",
    version="1.1.0"
)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    # "wav" is the original behaviour (still the default). "mp3" returns a much smaller file.
    format: Literal["wav", "mp3"] = "wav"


@lru_cache(maxsize=1)
def get_voice():
    if not MODEL_PATH.exists():
        raise RuntimeError("Piper voice model not found.")
    return PiperVoice.load(str(MODEL_PATH))


def clean_key(value: str | None) -> str:
    """Keep only visible ASCII characters. This drops spaces, new lines, zero-width characters and a BOM,
    which sneak in easily when a key is copied and pasted on a phone, plus any quotes around the key."""
    return re.sub(r"[^\x21-\x7E]", "", value or "").strip("\"'")


def verify_api_key(api_key: str | None):
    expected_key = clean_key(os.getenv("PIPER_API_KEY"))
    api_key = clean_key(api_key)
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured."
        )
    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )


def wav_to_mp3(wav_bytes: bytes, bitrate_kbps: int = 64) -> bytes:
    """Re-encode Piper's WAV output as MP3 (speech sounds fine at 64 kbps; about 1/5 of the size)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        pcm = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError("Expected 16-bit audio from Piper.")

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(channels)
    encoder.set_quality(2)
    return bytes(encoder.encode(pcm) + encoder.flush())


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

        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            voice.synthesize_wav(request.text, wav_file)
        wav_bytes = audio_buffer.getvalue()

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

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {str(error)}"
        )
