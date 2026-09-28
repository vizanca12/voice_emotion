"""Inference server: FastAPI + the recording page.

Audio arrives as raw float32 PCM already resampled to 16 kHz by the browser
(Web Audio API). That keeps ffmpeg off the server entirely -- the deployed
container would otherwise need it just to decode the WebM that MediaRecorder
produces, and the free Hugging Face image has no system packages.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C                                    # noqa: E402
from src.features import LogMel, fix_length                    # noqa: E402
from src.models import MODEL_DESCRIPTIONS, build_model         # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Voice Emotion Recognition", version="1.0")

_frontend = LogMel().eval()
_models: dict[str, dict] = {}


def load_models() -> None:
    """Load every checkpoint present. A missing model is skipped, not fatal:
    the page then simply shows fewer columns."""
    for ckpt_path in sorted(C.CKPT_DIR.glob("*.pt")):
        try:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            name = blob["model_name"]
            kwargs = {"pretrained": False} if name.startswith("transfer") else {}
            model = build_model(name, **kwargs)
            model.load_state_dict(blob["state_dict"])
            model.eval()
            _models[name] = {
                "model": model,
                "metrics": blob.get("test_metrics", {}),
                "description": MODEL_DESCRIPTIONS.get(name, name),
            }
            print(f"loaded {name} from {ckpt_path.name}")
        except Exception as exc:                                # noqa: BLE001
            print(f"could not load {ckpt_path.name}: {exc}")
    if not _models:
        print("WARNING: no checkpoints in checkpoints/ -- run `python -m src.train --all` first")


load_models()


# --------------------------------------------------------------------------- helpers


def encode_spectrogram(spec: torch.Tensor) -> dict:
    """Quantise the mel image to uint8 + base64.

    The raw 64x301 float array is ~100 kB of JSON per request; as bytes it is
    19 kB, which matters on a free CPU instance.
    """
    arr = spec.detach().cpu().numpy().astype(np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    scale = max(vmax - vmin, 1e-6)
    q = np.clip((arr - vmin) / scale * 255.0, 0, 255).astype(np.uint8)
    return {
        "data": base64.b64encode(q.tobytes()).decode("ascii"),
        "n_mels": int(arr.shape[0]),
        "n_frames": int(arr.shape[1]),
        "vmin": vmin,
        "vmax": vmax,
    }


@torch.no_grad()
def infer(wav: torch.Tensor, sample_rate: int) -> dict:
    if wav.dim() > 1:
        wav = wav.mean(dim=0)
    if wav.numel() < 1024:
        raise HTTPException(400, "recording too short")

    if sample_rate != C.SAMPLE_RATE:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sample_rate, C.SAMPLE_RATE)

    duration = wav.numel() / C.SAMPLE_RATE
    spec = _frontend(fix_length(wav))                 # (1, 1, n_mels, n_frames)

    predictions = {}
    for name, entry in _models.items():
        probs = F.softmax(entry["model"](spec), dim=1)[0]
        order = torch.argsort(probs, descending=True)
        predictions[name] = {
            "probs": {C.EMOTIONS[i]: round(float(probs[i]), 4) for i in range(len(C.EMOTIONS))},
            "top": C.EMOTIONS[int(order[0])],
            "confidence": round(float(probs[order[0]]), 4),
            "runner_up": C.EMOTIONS[int(order[1])],
        }

    # A coarse loudness envelope, so the page can draw the waveform it analysed.
    frames = fix_length(wav).unfold(0, C.HOP_LENGTH, C.HOP_LENGTH)
    envelope = frames.abs().max(dim=1).values
    envelope = (envelope / envelope.max().clamp(min=1e-6)).round(decimals=3)

    return {
        "duration": round(duration, 2),
        "spectrogram": encode_spectrogram(spec[0, 0]),
        "envelope": envelope.tolist(),
        "predictions": predictions,
        "emotions": C.EMOTIONS,
    }


# --------------------------------------------------------------------------- routes


@app.get("/api/meta")
def meta() -> JSONResponse:
    return JSONResponse({
        "emotions": C.EMOTIONS,
        "models": [
            {
                "name": name,
                "description": entry["description"],
                "uar": entry["metrics"].get("uar"),
                "accuracy": entry["metrics"].get("accuracy"),
            }
            for name, entry in _models.items()
        ],
        "audio": {
            "sample_rate": C.SAMPLE_RATE,
            "clip_seconds": C.CLIP_SECONDS,
            "n_mels": C.N_MELS,
            "n_fft": C.N_FFT,
            "hop_length": C.HOP_LENGTH,
        },
    })


@app.post("/api/predict")
async def predict(request: Request, sr: int = C.SAMPLE_RATE) -> JSONResponse:
    """Raw little-endian float32 mono PCM in the request body."""
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "empty body")
    wav = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())
    return JSONResponse(infer(wav, sr))


@app.post("/api/predict-file")
async def predict_file(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a wav/flac/ogg instead -- handy for curl and for the test suite."""
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(await file.read()), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T.copy())
    return JSONResponse(infer(wav, sample_rate))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": list(_models)}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
