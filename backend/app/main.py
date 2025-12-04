from io import BytesIO

import torch
from torchvision.transforms import v2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.dependencies import generate_caption
from app.config import TRANSFORM_IMAGE_SIZE

app = FastAPI(
    title="Image Whisper API.",
    description="Caption an image using a modern transformer-based model.",
    version="1.0",
)

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(TRANSFORM_IMAGE_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok"}


@app.post("/caption", tags=["Caption"])
async def caption(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        captions = generate_caption(tensor)

        return {"captions": captions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
