"""NSFW and weapons detection validator."""
import sys
sys.path.append('..')
from config.settings import Config


class NSFWValidator:
    @staticmethod
    def validate(image_path):
        # Lazy import to avoid loading models at import time
        from models.manager import model_manager
        """Check for NSFW content and weapons"""
        try:
            # Ensure models are loaded
            if not model_manager.models_loaded:
                if not model_manager.load_models():
                    raise Exception("Failed to load required models")
            
            # Check if detectors are available
            if model_manager.nsfw_detector is None:
                raise Exception("NSFW detector not available")
            if model_manager.weapons_detector is None:
                raise Exception("Weapons detector not available")
            
            # NSFW Detection
            nsfw_predictions = model_manager.nsfw_detector(image_path)
            nsfw_score = max([
                pred['score'] for pred in nsfw_predictions 
                if 'nsfw' in pred['label'].lower() or 'explicit' in pred['label'].lower()
            ] + [0.0])
            is_nsfw = nsfw_score > Config.NSFW_THRESHOLD
            
            # Weapons Detection - Updated for NabilaLM model labels
            weapons_predictions = model_manager.weapons_detector(image_path)
            print(f"Weapons predictions: {weapons_predictions}")
            
            # Map generic labels to weapon categories for NabilaLM model
            weapon_label_mapping = {
                'label_1': {'category': 'firearms', 'name': 'gun'},
                'label_2': {'category': 'knives', 'name': 'knife'},
                'label_3': {'category': 'human', 'name': 'human being'},
                'LABEL_1': {'category': 'firearms', 'name': 'gun'},
                'LABEL_2': {'category': 'knives', 'name': 'knife'},
                'LABEL_3': {'category': 'human', 'name': 'human being'},
                # Add more mappings if needed
                'gun': {'category': 'firearms', 'name': 'gun'},
                'knife': {'category': 'knives', 'name': 'knife'},
                'pistol': {'category': 'firearms', 'name': 'pistol'},
                'rifle': {'category': 'firearms', 'name': 'rifle'},
                'weapon': {'category': 'firearms', 'name': 'weapon'},
                'blade': {'category': 'knives', 'name': 'blade'},
                'sword': {'category': 'knives', 'name': 'sword'},
                'bomb': {'category': 'firearms', 'name': 'bomb'},
                'explosive': {'category': 'firearms', 'name': 'explosive'},
                'grenade': {'category': 'firearms', 'name': 'grenade'}
            }
            
            detected_weapons = []
            highest_weapon_score = 0.0
            weapon_type = None
            
            for detection in weapons_predictions:
                if detection['score'] > Config.WEAPON_THRESHOLD:
                    label = detection['label']
                    
                    # Check if label matches our mapping
                    weapon_info = weapon_label_mapping.get(label)
                    if weapon_info:
                        # Skip if this is classified as human, not a weapon
                        if weapon_info['category'] == 'human':
                            print(f"Human detected (not weapon): {label} with score {detection['score']}")
                            continue
                            
                        detected_weapons.append({
                            'category': weapon_info['category'],
                            'label': weapon_info['name'],
                            'original_label': label,  # Keep original label for debugging
                            'score': detection['score'],
                            'box': detection['box']
                        })
                        
                        if detection['score'] > highest_weapon_score:
                            highest_weapon_score = detection['score']
                            weapon_type = weapon_info['category']
                    else:
                        # If label doesn't match mapping, still log it for debugging
                        print(f"Unknown weapon label detected: {label} with score {detection['score']}")
                        # Optionally, treat unknown labels as potential weapons
                        detected_weapons.append({
                            'category': 'unknown_weapon',
                            'label': f'unknown_{label}',
                            'original_label': label,
                            'score': detection['score'],
                            'box': detection['box']
                        })
                        
                        if detection['score'] > highest_weapon_score:
                            highest_weapon_score = detection['score']
                            weapon_type = 'unknown_weapon'
                    
            has_weapons = len(detected_weapons) > 0
            is_unsafe = is_nsfw or has_weapons
            
            # Generate reason
            if is_nsfw and has_weapons:
                reason = f'NSFW content and weapons detected'
            elif is_nsfw:
                reason = f'NSFW content detected (score: {nsfw_score:.3f})'
            elif has_weapons:
                weapon_names = [w['label'] for w in detected_weapons]
                reason = f'Weapons detected - {weapon_type}: {", ".join(weapon_names)} (highest score: {highest_weapon_score:.3f})'
            else:
                reason = 'Safe content - no NSFW or weapons detected'
            
            return {
                'is_nsfw': is_nsfw,
                'nsfw_score': nsfw_score,
                'has_weapons': has_weapons,
                'weapons_detected': detected_weapons,
                'weapon_count': len(detected_weapons),
                'highest_weapon_score': highest_weapon_score,
                'weapon_type': weapon_type,
                'is_unsafe': is_unsafe,
                'reason': reason,
                'nsfw_predictions': nsfw_predictions,
                'weapons_predictions': weapons_predictions
            }
            
        except Exception as e:
            print(f"Error in NSFW/weapons validation: {str(e)}")
            return {
                'is_nsfw': False, 'nsfw_score': 0.0, 'has_weapons': False,
                'weapons_detected': [], 'weapon_count': 0, 'highest_weapon_score': 0.0,
                'weapon_type': None, 'is_unsafe': False,
                'reason': f'Error in NSFW/weapons detection: {str(e)}',
                'nsfw_predictions': [], 'weapons_predictions': []
            }