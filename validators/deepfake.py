"""Deepfake detection validator."""
import sys
sys.path.append('..')
from config.settings import Config


class DeepfakeValidator:
    @staticmethod
    def validate(image_path):
        # Lazy import to avoid loading models at import time
        from models.manager import model_manager
        """Check for deepfake content"""
        try:
            # Ensure models are loaded
            if not model_manager.models_loaded:
                if not model_manager.load_models():
                    raise Exception("Failed to load required models")
            
            # Check if detector is available
            if model_manager.deepfake_detector is None:
                raise Exception("Deepfake detector not available")
            
            predictions = model_manager.deepfake_detector(image_path)
            deepfake_score = max([
                pred['score'] for pred in predictions 
                if any(keyword in pred['label'].lower() for keyword in ['fake', 'deepfake', 'ai'])
            ] + [0.0])
            
            is_deepfake = deepfake_score > Config.DEEPFAKE_THRESHOLD
            
            return {
                'is_deepfake': is_deepfake,
                'deepfake_score': deepfake_score,
                'reason': f'Deepfake detected (score: {deepfake_score:.3f})' if is_deepfake else 'Real image',
                'predictions': predictions
            }
        except Exception as e:
            return {
                'is_deepfake': False,
                'deepfake_score': 0.0,
                'reason': f'Error in deepfake detection: {str(e)}',
                'predictions': []
            }