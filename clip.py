from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("processed_matrimonial_photo.jpg").convert("RGB")

# candidate labels / prompts (tune these)
prompts = [
  "a picture showing self-harm",
  "a person's neck hanging from a noose",
  "a violent scene",
  "a weapon (gun, knife)",
  "a person standing normally",
  "a memorial or tribute image",
  "a nude image"
]

inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape (1, len(prompts))
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

for p, prob in zip(prompts, probs):
    print(f"{prob:.3f} — {p}")
