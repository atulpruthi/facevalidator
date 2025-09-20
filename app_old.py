
# Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline

img_path = r"C:\Users\91965\Downloads\8.jpg"
classifier = pipeline("object-detection", model="NabilaLM/detr-weapons-detection_40ep")
detections = classifier(img_path)
print(detections)
image = Image.open(img_path)
index = 0
for det in detections:
    box = det['box']
    cropped = image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
    cropped.save(r"C:\Users\91965\Downloads\cropped_{}.jpg".format(index))
    index += 1



#face_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
#"image-classification", model="tonyassi/celebrity-classifier"
#celebrity_detecttor = pipeline("image-classification", model="nateraw/vit-base-celebrity-faces")
#human_checker = pipeline("image-classification", model="microsoft/beit-base-patch16-224-pt22k")
#nsfw_detector = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
#quality_checker = pipeline("image-classification", model="microsoft/swin-base-patch4-window7-224")
#deepfake_detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
