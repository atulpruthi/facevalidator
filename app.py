from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
import io
import os

# Load model once at startup
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

# Load API key from environment variable
API_KEY = os.getenv("API_KEY", "changeme")

app = FastAPI(
    title="NSFW Detection API",
    description="REST API using Falconsai/nsfw_image_detection with API key protection",
    version="1.0.0",
)

def verify_api_key(key: str):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/")
def root(x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    return {"status": "ok", "message": "NSFW detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = classifier(image)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
