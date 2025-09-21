"""Model management for the matrimonial image validator API."""
import os
import pathlib
from pathlib import Path
from transformers import pipeline
from fastai.vision.all import load_learner
import sys
sys.path.append('..')
from config.settings import Config


class ModelManager:
    def __init__(self):
        self.face_detector = None
        self.nsfw_detector = None
        self.deepfake_detector = None
        self.weapons_detector = None
        self.celebrity_classifier = None
        self._models_loaded = False
    
    def load_models(self):
        """Load all required models"""
        try:
            print("Loading models...")
            self.face_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
            self.nsfw_detector = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
            self.deepfake_detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
            self.weapons_detector = pipeline("object-detection", model="NabilaLM/detr-weapons-detection_40ep")
            
            # Load celebrity classifier with fastai
            try:
                if os.path.exists(Config.CELEBRITY_MODEL_PATH):
                    # Fix for Windows path compatibility
                    temp = pathlib.PosixPath
                    pathlib.PosixPath = pathlib.WindowsPath
                    
                    # Method 1: Use load_learner (recommended)
                    try:
                        self.celebrity_classifier = load_learner(Config.CELEBRITY_MODEL_PATH)
                        print("Celebrity classifier loaded successfully with load_learner")
                    except:
                        # Method 2: Use load_learner with Path
                        self.celebrity_classifier = load_learner(Path(Config.CELEBRITY_MODEL_PATH))
                        print("Celebrity classifier loaded successfully with load_learner and Path")
                        
                    # Restore original pathlib
                    pathlib.PosixPath = temp
                else:
                    print(f"Warning: Celebrity classifier not found at {Config.CELEBRITY_MODEL_PATH}")
                    self.celebrity_classifier = None
            except Exception as celebrity_error:
                print(f"Warning: Failed to load celebrity classifier: {str(celebrity_error)}")
                print("Celebrity detection will be disabled. Other validations will continue to work.")
                self.celebrity_classifier = None
                
            self._models_loaded = True
            print("All models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    @property
    def models_loaded(self):
        # Core models are required, celebrity classifier is optional
        return self._models_loaded and all([
            self.face_detector, self.nsfw_detector, 
            self.deepfake_detector, self.weapons_detector
        ])


# Global model manager instance
model_manager = ModelManager()