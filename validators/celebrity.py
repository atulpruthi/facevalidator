"""Celebrity detection validator using fastai model."""
from PIL import Image
import sys
sys.path.append('..')
from config.settings import Config


class CelebrityValidator:
    @staticmethod
    def validate(image_path):
        """Wrapper for detect_celebrity method"""
        return CelebrityValidator.detect_celebrity(image_path)
    
    @staticmethod    
    def detect_celebrity(image_path):
        """Detect if image contains a celebrity face"""
        try:
            # Lazy import to avoid loading models at import time
            from models.manager import model_manager
            
            # Ensure models are loaded (but celebrity classifier is optional)
            if not model_manager._models_loaded:
                model_manager.load_models()
            
            if not model_manager.celebrity_classifier:
                return {
                    'is_celebrity': False,
                    'celebrity_name': None,
                    'confidence': 0.0,
                    'reason': 'Celebrity classifier not loaded'
                }
            
            # Load and preprocess image
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Make prediction using fastai
            try:
                prediction, pred_idx, probs = model_manager.celebrity_classifier.predict(image_path)
                
                # Get the best class and its probability
                best_class = model_manager.celebrity_classifier.dls.vocab[probs.argmax()]
                best_prob = probs.max().item()
                
                # Check if confidence is above threshold
                is_celebrity = best_prob > Config.CELEBRITY_THRESHOLD
                
                # Get celebrity name if detected
                celebrity_name = str(best_class) if is_celebrity else None
                
                return {
                    'is_celebrity': is_celebrity,
                    'celebrity_name': celebrity_name,
                    'confidence': float(best_prob),
                    'reason': f'Celebrity detected: {celebrity_name}' if is_celebrity else 'No celebrity detected'
                }
                
            except Exception as model_error:
                print(f"Error in celebrity model prediction: {str(model_error)}")
                return {
                    'is_celebrity': False,
                    'celebrity_name': None,
                    'confidence': 0.0,
                    'reason': f'Model prediction error: {str(model_error)}'
                }
                
        except Exception as e:
            return {
                'is_celebrity': False,
                'celebrity_name': None,
                'confidence': 0.0,
                'reason': f'Error in celebrity detection: {str(e)}'
            }