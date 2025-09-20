from PIL import Image
from fastai.vision.all import *
import pathlib

# Try loading with proper Learner.load syntax
try:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    # Method 1: Use load_learner (recommended)
    classifier = load_learner("./models/celebrity-classifier.pkl")
    print("Model loaded successfully with load_learner")
except:
    try:
        # Method 2: Use load_learner with Path
        classifier = load_learner(Path("./models/celebrity-classifier.pkl"))
        print("Model loaded successfully with load_learner and Path")
    except Exception as e:
        print(f"Error loading model with load_learner: {e}")
        exit(1)

# Load and preprocess image
img = Image.open(r"C:\Users\91965\Downloads\1000113221.jpg")

# Convert to RGB if needed
if img.mode != 'RGB':
    img = img.convert('RGB')

# Make prediction
try:
    prediction, pred_idx, probs = classifier.predict(r"C:\Users\91965\Downloads\1000113221.jpg")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {probs[pred_idx]:.4f}")
    best_class = classifier.dls.vocab[probs.argmax()]
    best_prob = probs.max().item()

    print(f"Best prediction: {best_class} ({best_prob:.2%})")
except Exception as e:
    print(f"Error during prediction: {e}")