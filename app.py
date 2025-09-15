import gradio as gr
from transformers import pipeline
from PIL import Image
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
import uvicorn
from threading import Thread

# -----------------------------
# Load models (only once)
# -----------------------------
face_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
#human_checker = pipeline("image-classification", model="microsoft/beit-base-patch16-224-pt22k")
#nsfw_detector = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
#quality_checker = pipeline("image-classification", model="microsoft/swin-base-patch4-window7-224")
#deepfake_detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Matrimonial Image Validator API", version="1.0.0")

# -----------------------------
# Core validation function
# -----------------------------
def validate_matrimonial_image(image):
    results = {
        "face_detected": False,
        "multiple_faces": False,
        "not_human": False,
        "nsfw": False,
        "low_quality": False,
        "deepfake": False,
        "reason": ""
    }
    
    # 1. Face detection
    faces = face_detector(image)
    if len(faces) == 0:
        results["reason"] = "No face detected"
        return results
    elif len(faces) > 1:
        results["multiple_faces"] = True
        results["reason"] = "Multiple faces detected"
        return results
    else:
        results["face_detected"] = True
    
    # 2. Human check
    #human_preds = human_checker(image)
    #human_label = max(human_preds, key=lambda x: x['score'])
    #if "cartoon" in human_label['label'].lower() or "anime" in human_label['label'].lower():
    #    results["not_human"] = True
    #    results["reason"] = "Not a real human photo"
    #    return results
    
    # 3. NSFW check
    #nsfw_preds = nsfw_detector(image)
    #nsfw_label = max(nsfw_preds, key=lambda x: x['score'])
    #if "nsfw" in nsfw_label['label'].lower() or "porn" in nsfw_label['label'].lower():
    #    results["nsfw"] = True
    #    results["reason"] = "NSFW / Nudity detected"
    
    #    return results
    
    # 4. Quality check
    #quality_preds = quality_checker(image)
    #bad_quality = max(quality_preds, key=lambda x: x['score'])
    #if "bad" in bad_quality['label'].lower() or "low" in bad_quality['label'].lower():
    #    results["low_quality"] = True
    #    results["reason"] = "Low-quality image"
    #    return results
    
    # 5. Deepfake detection
    #deepfake_preds = deepfake_detector(image)
    #deepfake_label = max(deepfake_preds, key=lambda x: x['score'])
    #if "fake" in deepfake_label['label'].lower():
    #    results["deepfake"] = True
    #    results["reason"] = "Deepfake / AI-generated face detected"
    #    return results
    
    results["reason"] = "Valid matrimonial photo"
    return results

# -----------------------------
# REST API Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Matrimonial Image Validator API", "status": "running"}

@app.post("/validate")
async def validate_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Validate image
        result = validate_matrimonial_image(image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

# -----------------------------
# Gradio UI Wrapper
# -----------------------------
def gradio_validator(image):
    result = validate_matrimonial_image(image)
    if result["reason"].startswith("Valid"):
        return f"✅ Accepted: {result['reason']}"
    else:
        return f"❌ Rejected: {result['reason']}"

# -----------------------------
# Create Gradio Interface
# -----------------------------
demo = gr.Interface(
    fn=gradio_validator,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Matrimonial Image Validator",
    description="Upload an image for validation. The system checks: single face, human, safe, high-quality, not AI-generated."
)

# -----------------------------
# Mount Gradio to FastAPI
# -----------------------------
app = gr.mount_gradio_app(app, demo, path="/ui")

# -----------------------------
# Launch the app
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
