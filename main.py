import io
import os
import wave
import secrets
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from piper import PiperVoice


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "en_GB-alan-medium.onnx"

app = FastAPI(
    title="Self-Hosted Piper TTS API",
    description="Text-to-speech API powered by Piper.",
    version="1.0.0"
)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@lru_cache(maxsize=1)
def get_voice():
    if not MODEL_PATH.exists():
        raise RuntimeError("Piper voice model not found.")

    return PiperVoice.load(str(MODEL_PATH))


def verify_api_key(api_key: str | None):
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


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "Piper TTS API",
        "model_available": MODEL_PATH.exists()
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

        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech.wav"'
            }
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {str(error)}"
        )
