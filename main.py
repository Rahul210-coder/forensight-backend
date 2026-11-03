import base64
import io
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI(title="ForenSight AI Backend", version="1.0")

# Enable Cross-Origin Resource Sharing for your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "best_forensic_densenet_effnetv2.pth")
CLASSES = ["real", "manipulated", "ai_generated"]

# Dual-Stream Architecture: DenseNet-121 (RGB) + EfficientNet-V2-S (ELA)
class ForenSightDualStream(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.rgb_backbone = models.densenet121(weights=None)
        rgb_dim = self.rgb_backbone.classifier.in_features  # 1024
        self.rgb_backbone.classifier = nn.Identity()

        self.ela_backbone = models.efficientnet_v2_s(weights=None)
        ela_dim = self.ela_backbone.classifier[1].in_features  # 1280
        self.ela_backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(rgb_dim + ela_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, ela):
        f_rgb = self.rgb_backbone(rgb)
        f_ela = self.ela_backbone(ela)
        return self.classifier(torch.cat((f_rgb, f_ela), dim=1))

model = ForenSightDualStream(num_classes=3).to(device)
is_loaded = False

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        is_loaded = True
        print(f"[INFO] Weights loaded successfully from '{MODEL_PATH}'.")
    except Exception as e:
        print(f"[ERROR] Could not load checkpoint: {e}")
else:
    print(f"[WARNING] Weights file '{MODEL_PATH}' not found in runtime directory.")

def compute_ela(pil_img: Image.Image, quality: int = 90) -> Image.Image:
    buffer = io.BytesIO()
    pil_img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    diff = ImageChops.difference(pil_img, compressed)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    return ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)

norm_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

norm_ela = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

@app.get("/health")
def health():
    return {
        "status": "online",
        "model_loaded": is_loaded,
        "device": str(device),
        "model": "DenseNet-121 + EfficientNet-V2-S",
        "benchmark_val_acc": 91.33
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image.")

    contents = await file.read()
    raw_img = Image.open(io.BytesIO(contents)).convert("RGB")
    ela_img = compute_ela(raw_img, quality=90)

    rgb_tensor = norm_rgb(raw_img).unsqueeze(0).to(device)
    ela_tensor = norm_ela(ela_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(rgb_tensor, ela_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    return {
        "verdict": CLASSES[idx],
        "confidence": round(float(probs[idx]) * 100.0, 2),
        "probabilities": {
            "real": round(float(probs[0]) * 100.0, 2),
            "manipulated": round(float(probs[1]) * 100.0, 2),
            "ai_generated": round(float(probs[2]) * 100.0, 2)
        },
        "ela_image": pil_to_base64(ela_img),
        "model_info": {
            "architecture": "DenseNet-121 + EfficientNet-V2-S",
            "benchmark_val_acc": 91.33
        }
    }