# app.py
import io
import os
import base64
import logging
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

import torch
from diffusers import StableDiffusionPipeline


# Logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepfake-api")


# FastAPI app

app = FastAPI(title="DeepFake Detection & Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Detection model 

MODEL_PATH = os.environ.get("MODEL_PATH", "deepMODEmain.h5")
IMG_TARGET_SIZE = (224, 224)
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))


# Load detection model 

try:
    model = load_model(MODEL_PATH)
    logger.info(f"Loaded Keras model from {MODEL_PATH}")
except Exception:
    logger.exception("Failed to load detection model")
    model = None


# Load Stable Diffusion 

logger.info(" Loading Stable Diffusion 1.5 (CPU, quality mode)...")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
)

pipe = pipe.to("cpu")

logger.info("✅ Stable Diffusion loaded")


# Pydantic models

class ImageRequest(BaseModel):
    image_base64: str


class PredictResponse(BaseModel):
    label: str
    score: float
    meta: Optional[Dict[str, Any]] = None


class GenerateImageRequest(BaseModel):
    prompt: str


class GenerateImageResponse(BaseModel):
    image_base64: str


# Detection helpers (UNCHANGED)

def preprocess_image_base64(b64: str, target_size=IMG_TARGET_SIZE):
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def decode_prediction(pred_arr: np.ndarray, threshold=THRESHOLD):
    p = float(np.squeeze(pred_arr))
    label = "FAKE" if p >= threshold else "REAL"
    return label, p


# Generation helper (LOCAL, HIGH QUALITY)

def generate_image_from_prompt(prompt: str) -> bytes:
    image = pipe(
        prompt,
        num_inference_steps=40,   # QUALITY
        guidance_scale=7.5,
        height=512,
        width=512,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# Routes

@app.get("/")
def root():
    return {
        "status": "ok",
        "detection_model_loaded": model is not None,
        "generation_backend": "local-stable-diffusion",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "detection_model_loaded": model is not None,
        "generation_ready": True,
    }

# ---------------- DETECTION MODE (UNCHANGED) ----------------
@app.post("/predict_image", response_model=PredictResponse)
async def predict_image(req: ImageRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    x = preprocess_image_base64(req.image_base64)
    preds = model.predict(x)
    label, score = decode_prediction(preds)

    return PredictResponse(
        label=label,
        score=score,
        meta={"model": os.path.basename(MODEL_PATH)},
    )


@app.post("/predict")
async def predict_auto(payload: dict):
    return await predict_image(
        ImageRequest(image_base64=payload["image_base64"])
    )

# ---------------- GENERATION MODE ----------------
@app.post("/generate/image", response_model=GenerateImageResponse)
def generate_image(req: GenerateImageRequest):
    logger.info("➡️ /generate/image called")
    logger.info(f"➡️ Prompt: {req.prompt}")

    image_bytes = generate_image_from_prompt(req.prompt)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {"image_base64": image_base64}


# Run server

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

