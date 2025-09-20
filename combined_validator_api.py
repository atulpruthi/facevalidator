from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter
import os
import tempfile
import io
import base64
import math
from datetime import datetime, timedelta
from functools import wraps

import jwt
import numpy as np
import cv2
import mediapipe as mp
from transformers import pipeline
import logging

# ==================== Configuration ====================
class Config:
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'temp_uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'yl_poiuytrewqasdfghjkl')
    JWT_ALGORITHM = 'HS256'
    
    # Model configurations
    NSFW_THRESHOLD = 0.7
    DEEPFAKE_THRESHOLD = 0.7
    WEAPON_THRESHOLD = 0.5
    FACE_CONFIDENCE_THRESHOLD = 0.7
    PERSON_DETECTION_THRESHOLD = 0.8
    MIN_PERSON_AREA = 5000

# ==================== Application Setup ====================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# ==================== Model Management ====================
class ModelManager:
    def __init__(self):
        self.face_detector = None
        self.nsfw_detector = None
        self.deepfake_detector = None
        self.weapons_detector = None
        self._models_loaded = False
    
    def load_models(self):
        """Load all required models"""
        try:
            print("Loading models...")
            self.face_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
            self.nsfw_detector = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
            self.deepfake_detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
            self.weapons_detector = pipeline("object-detection", model="NabilaLM/detr-weapons-detection_40ep")
            self._models_loaded = True
            print("All models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    @property
    def models_loaded(self):
        return self._models_loaded and all([
            self.face_detector, self.nsfw_detector, 
            self.deepfake_detector, self.weapons_detector
        ])

model_manager = ModelManager()

# ==================== Utilities ====================
class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def pil_to_base64(pil_image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    return base64.b64encode(buffer.getvalue()).decode()

def process_uploaded_file(file_data, filename):
    """Process uploaded file from FormData"""
    try:
        # Handle both file objects and buffers
        if hasattr(file_data, 'read'):
            file_content = file_data.read()
        else:
            file_content = file_data
        
        # Create temporary file
        file_ext = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(file_content)
            return temp_file.name
            
    except Exception as e:
        raise ValidationError(f"Error processing uploaded file: {str(e)}")

# ==================== Authentication ====================
class AuthManager:
    @staticmethod
    def validate_token(token):
        """Validate JWT token"""
        try:
            logger.debug(f"Validating token: {token[:20]}..." if token else "None")
            
            if not token:
                return False, "No token provided"
            
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
                logger.debug("Removed Bearer prefix")
            
            logger.debug(f"JWT_SECRET_KEY: {Config.JWT_SECRET_KEY}")
            logger.debug(f"JWT_ALGORITHM: {Config.JWT_ALGORITHM}")
            
            # Decode and validate the JWT token
            payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
            logger.debug(f"Token payload: {payload}")
            
            # Check expiration
            if 'exp' in payload:
                exp_time = datetime.fromtimestamp(payload['exp'])
                current_time = datetime.utcnow()
                logger.debug(f"Token expires: {exp_time}, Current time: {current_time}")
                
                if current_time.timestamp() > payload['exp']:
                    return False, "Token has expired"
            
            # Check required claims
            #if 'user_id' not in payload:
            #    return False, "Missing required claim: user_id"
            
            return True, payload
            
        except jwt.ExpiredSignatureError:
            logger.error("Token has expired")
            return False, "Token has expired"
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {str(e)}")
            return False, f"Invalid token: {str(e)}"
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            return False, f"Token validation error: {str(e)}"
    
    @staticmethod
    def generate_token(user_id, expires_in_hours=24):
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=expires_in_hours)
        }
        return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)

def require_auth(f):
    """Decorator for JWT authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        logger.debug(f"Auth header received: {auth_header}")
        
        if not auth_header:
            logger.warning("Missing Authorization header")
            return jsonify({'error': 'Missing Authorization header'}), 401
        
        is_valid, result = AuthManager.validate_token(auth_header)
        logger.debug(f"Token validation result: valid={is_valid}, result={result}")
        
        if not is_valid:
            logger.warning(f"Authentication failed: {result}")
            return jsonify({'error': f'Authentication failed: {result}'}), 401
        
        request.user = result
        logger.debug(f"Authentication successful for user: {result.get('user_id')}")
        return f(*args, **kwargs)
    return decorated_function

# ==================== Validation Services ====================
class NSFWValidator:
    @staticmethod
    def validate(image_path):
        """Check for NSFW content and weapons"""
        try:
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
                'bomb': {'category': 'bombs', 'name': 'bomb'},
                'explosive': {'category': 'bombs', 'name': 'explosive'},
                'grenade': {'category': 'bombs', 'name': 'grenade'}
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

class DeepfakeValidator:
    @staticmethod
    def validate(image_path):
        """Check for deepfake content"""
        try:
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

class ImageProcessor:
    @staticmethod
    def get_sorted_boxes(detections):
        """Sort detection boxes by area"""
        boxes_with_area = []
        for det in detections:
            box = det['box']
            area = (box['xmax'] - box['xmin']) * (box['ymax'] - box['ymin'])
            boxes_with_area.append((det, area))
        return sorted(boxes_with_area, key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def create_person_mask(image_path, box):
        """Create person segmentation mask"""
        try:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            original_image = Image.open(image_path)
            original_array = np.array(original_image.convert('RGB'))
            
            with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
                results = selfie_segmentation.process(original_array)
                mask = results.segmentation_mask
                return mask[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
        except Exception as e:
            print(f"Error creating person mask: {e}")
            return None
    
    @staticmethod
    def get_cropped_image_with_blur(image_path, box, blur_radius=15):
        """Crop image and blur background"""
        try:
            original_image = Image.open(image_path)
            cropped_image = original_image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
            
            person_mask = ImageProcessor.create_person_mask(image_path, box)
            
            if person_mask is not None:
                mask_pil = Image.fromarray((person_mask * 255).astype(np.uint8), mode='L')
                mask_pil = mask_pil.resize(cropped_image.size, Image.LANCZOS)
                blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                
                mask_array = np.array(mask_pil)
                binary_mask = (mask_array > 128).astype(np.uint8) * 255
                binary_mask_pil = Image.fromarray(binary_mask, mode='L')
                
                return Image.composite(cropped_image, blurred_image, binary_mask_pil)
            else:
                return cropped_image
                
        except Exception as e:
            print(f"Error in background blur: {e}")
            return Image.open(image_path).crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))

class PoseValidator:
    @staticmethod
    def validate_objects(image_path):
        """Detect and validate person objects"""
        try:
            detections = model_manager.face_detector(image_path)
            persons = [det for det in detections if det['label'] == 'person' and det['score'] > Config.PERSON_DETECTION_THRESHOLD]
            
            boxes_with_area = ImageProcessor.get_sorted_boxes(persons)
            
            if len(boxes_with_area) == 0:
                return {'reason': "No persons detected", 'has_single_person': False}
            
            if len(boxes_with_area) == 1:
                largest_box, largest_area = boxes_with_area[0]
                if largest_area < Config.MIN_PERSON_AREA:
                    return {'reason': "Small person detected", 'has_single_person': False, 'person_area': largest_area}
                else:
                    cropped_image = ImageProcessor.get_cropped_image_with_blur(image_path, largest_box['box'])
                    return {'image': cropped_image, 'reason': "Valid single person", 'has_single_person': True, 'person_area': largest_area}
            
            # Multiple persons detected
            largest_box, largest_area = boxes_with_area[0]
            second_largest_box, second_largest_area = boxes_with_area[1]
            percentage_diff = ((largest_area - second_largest_area) / second_largest_area) * 100 if second_largest_area > 0 else float('inf')
            
            if percentage_diff < 200:
                return {'reason': "Multiple persons detected", 'has_single_person': False, 'person_count': len(boxes_with_area)}
            else:
                cropped_image = ImageProcessor.get_cropped_image_with_blur(image_path, largest_box['box'])
                return {'image': cropped_image, 'reason': "Dominant single person", 'has_single_person': True, 'person_area': largest_area}
                
        except Exception as e:
            return {'reason': f"Error in object detection: {str(e)}", 'has_single_person': False}
    
    @staticmethod
    def detect_frontal_face(pil_image):
        """Detect frontal faces using MediaPipe"""
        try:
            mp_face_detection = mp.solutions.face_detection
            rgb_array = np.array(pil_image.convert('RGB'))
            
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(rgb_array)
                
                if not results.detections:
                    return {'frontal_face_detected': False, 'face_count': 0, 'reason': 'No faces detected'}
                
                if len(results.detections) > 1:
                    return {'frontal_face_detected': False, 'face_count': len(results.detections), 'reason': 'Multiple faces detected'}
                
                detection = results.detections[0]
                confidence = detection.score[0]
                is_frontal = confidence > Config.FACE_CONFIDENCE_THRESHOLD
                
                return {
                    'frontal_face_detected': is_frontal,
                    'face_count': 1,
                    'confidence': confidence,
                    'reason': f'Frontal face detected (confidence: {confidence:.2f})' if is_frontal else 'Low confidence frontal face'
                }
                
        except Exception as e:
            return {'frontal_face_detected': False, 'face_count': 0, 'reason': f'Error in face detection: {str(e)}'}
    
    @staticmethod
    def detect_pose_orientation(pil_image):
        """Detect frontal pose using MediaPipe Pose"""
        try:
            mp_pose = mp.solutions.pose
            rgb_array = np.array(pil_image.convert('RGB'))
            
            with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
                results = pose.process(rgb_array)
                
                if not results.pose_landmarks:
                    return {'frontal_pose': False, 'reason': 'No pose landmarks detected', 'pose_score': 0.0}
                
                landmarks = results.pose_landmarks.landmark
                h, w = rgb_array.shape[:2]
                
                # Extract key landmarks
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
                left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
                right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
                
                # Calculate metrics
                both_eyes_visible = left_eye.visibility > 0.5 and right_eye.visibility > 0.5
                both_shoulders_visible = left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5
                
                if both_shoulders_visible:
                    shoulder_distance = abs((left_shoulder.x - right_shoulder.x) * w)
                    shoulder_width_ratio = shoulder_distance / w
                    nose_offset = abs(nose.x - (left_shoulder.x + right_shoulder.x) / 2)
                else:
                    shoulder_width_ratio = 0.0
                    nose_offset = 1.0
                
                if both_eyes_visible:
                    eye_distance = abs((left_eye.x - right_eye.x) * w)
                    eye_distance_ratio = eye_distance / w
                    eye_center_x = (left_eye.x + right_eye.x) / 2
                    nose_eye_offset = abs(nose.x - eye_center_x)
                else:
                    eye_distance_ratio = 0.0
                    nose_eye_offset = 1.0
                
                ear_visibility_ratio = (left_ear.visibility + right_ear.visibility) / 2
                eye_visibility_ratio = (left_eye.visibility + right_eye.visibility) / 2
                eye_visibility_balance = abs(left_eye.visibility - right_eye.visibility)
                
                # Frontal pose criteria
                is_frontal = (
                    both_shoulders_visible and both_eyes_visible and
                    shoulder_width_ratio > 0.15 and
                    eye_distance_ratio > 0.08 and eye_distance_ratio < 0.35 and
                    nose_offset < 0.1 and nose_eye_offset < 0.08 and
                    ear_visibility_ratio > 0.3 and
                    eye_visibility_ratio > 0.6 and
                    eye_visibility_balance < 0.4
                )
                
                pose_score = min(1.0, (
                    shoulder_width_ratio * 1.5 + eye_distance_ratio * 3 +
                    (1 - nose_offset) * 2 + (1 - nose_eye_offset) * 2 +
                    ear_visibility_ratio * 1.5 + eye_visibility_ratio * 2 +
                    (1 - eye_visibility_balance) * 1.5
                ) / 11)
                
                # Generate reason
                if not is_frontal:
                    if not both_eyes_visible:
                        reason = f"Side pose - eye visibility issue (L:{left_eye.visibility:.2f}, R:{right_eye.visibility:.2f})"
                    elif not both_shoulders_visible:
                        reason = "Side pose - shoulder visibility issue"
                    elif shoulder_width_ratio <= 0.15:
                        reason = f"Side pose - narrow shoulders ({shoulder_width_ratio:.2f})"
                    elif nose_offset >= 0.1:
                        reason = f"Side pose - nose not centered ({nose_offset:.2f})"
                    else:
                        reason = "Side pose detected"
                else:
                    reason = f"Frontal pose detected (score: {pose_score:.2f})"
                
                return {
                    'frontal_pose': is_frontal,
                    'reason': reason,
                    'pose_score': pose_score,
                    'metrics': {
                        'shoulder_width_ratio': shoulder_width_ratio,
                        'eye_distance_ratio': eye_distance_ratio,
                        'nose_offset': nose_offset,
                        'eye_visibility_ratio': eye_visibility_ratio,
                        'both_eyes_visible': both_eyes_visible
                    }
                }
                
        except Exception as e:
            return {'frontal_pose': False, 'reason': f'Error in pose detection: {str(e)}', 'pose_score': 0.0}
    
    @staticmethod
    def validate(image_path):
        """Complete pose validation pipeline"""
        try:
            # Step 1: Object detection
            object_result = PoseValidator.validate_objects(image_path)
            if not object_result['has_single_person']:
                return {
                    'valid_pose': False, 'reason': object_result['reason'],
                    'has_single_person': False, 'frontal_face': False, 'frontal_pose': False
                }
            
            img = object_result.get('image')
            if img is None:
                return {
                    'valid_pose': False, 'reason': 'No valid cropped image',
                    'has_single_person': True, 'frontal_face': False, 'frontal_pose': False
                }
            
            # Step 2: Face detection
            face_result = PoseValidator.detect_frontal_face(img)
            if not face_result['frontal_face_detected']:
                return {
                    'valid_pose': False, 'reason': f"Face issue: {face_result['reason']}",
                    'has_single_person': True, 'frontal_face': False, 'frontal_pose': False,
                    'face_confidence': face_result.get('confidence', 0.0),
                    'processed_image': pil_to_base64(img)
                }
            
            # Step 3: Pose detection
            pose_result = PoseValidator.detect_pose_orientation(img)
            if not pose_result['frontal_pose']:
                return {
                    'valid_pose': False, 'reason': f"Pose issue: {pose_result['reason']}",
                    'has_single_person': True, 'frontal_face': True, 'frontal_pose': False,
                    'face_confidence': face_result['confidence'],
                    'pose_score': pose_result['pose_score'],
                    'pose_metrics': pose_result.get('metrics', {}),
                    'processed_image': pil_to_base64(img)
                }
            
            # All checks passed
            return {
                'valid_pose': True,
                'reason': 'Valid matrimonial photo - frontal face and pose',
                'has_single_person': True, 'frontal_face': True, 'frontal_pose': True,
                'face_confidence': face_result['confidence'],
                'pose_score': pose_result['pose_score'],
                'pose_metrics': pose_result['metrics'],
                'processed_image': pil_to_base64(img)
            }
            
        except Exception as e:
            return {
                'valid_pose': False, 'reason': f'Error in pose validation: {str(e)}',
                'has_single_person': False, 'frontal_face': False, 'frontal_pose': False
            }

# ==================== Flask Routes ====================
@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    return jsonify({
        'name': 'Matrimonial Image Validator API',
        'version': '2.0.0',
        'description': 'Comprehensive image validation for matrimonial profiles',
        'authentication': 'JWT Bearer token required',
        'endpoints': {
            'POST /validate': 'Complete validation pipeline',
            'POST /validate/nsfw': 'NSFW and weapons detection',
            'POST /validate/deepfake': 'Deepfake detection',
            'POST /validate/pose': 'Pose validation',
            'POST /auth/token': 'Generate test token',
            'GET /health': 'Health check'
        },
        'models_loaded': model_manager.models_loaded
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'face_detector': model_manager.face_detector is not None,
            'nsfw_detector': model_manager.nsfw_detector is not None,
            'deepfake_detector': model_manager.deepfake_detector is not None,
            'weapons_detector': model_manager.weapons_detector is not None
        }
    })

@app.route('/auth/token', methods=['POST'])
def generate_test_token():
    """Generate test JWT token"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'test_user')
        expires_in_hours = data.get('expires_in_hours', 24)
        
        token = AuthManager.generate_token(user_id, expires_in_hours)
        
        return jsonify({
            'token': token,
            'user_id': user_id,
            'expires_in_hours': expires_in_hours,
            'token_type': 'Bearer'
        })
    except Exception as e:
        return jsonify({'error': f'Token generation failed: {str(e)}'}), 500

def handle_file_upload():
    """Common file upload handling"""
    file_data = request.files.get('file') or request.files.get('image')
    if not file_data or file_data.filename == '':
        raise ValidationError('No file provided')
    
    if not allowed_file(file_data.filename):
        raise ValidationError('Invalid file type. Allowed: png, jpg, jpeg, gif, bmp')
    
    return process_uploaded_file(file_data, file_data.filename)

@app.route('/validate', methods=['POST'])
@require_auth
def validate_complete():
    """Complete validation pipeline"""
    temp_path = None
    try:
        temp_path = handle_file_upload()
        user_id = request.user.get('user_id')
        
        # Step 1: NSFW/Weapons validation
        nsfw_result = NSFWValidator.validate(temp_path)
        if nsfw_result['is_unsafe']:
            return jsonify({
                'valid': False, 'reason': "Please do not upload offensive/nudity content.",
                'stage': 'nsfw_weapons_detection',
                'nsfw_result': nsfw_result, 'user_id': user_id
            })
        
        # Step 2: Deepfake validation
        deepfake_result = DeepfakeValidator.validate(temp_path)
        if deepfake_result['is_deepfake']:
            return jsonify({
                'valid': False, 'reason': "Please upload a real photo, deepfake images are not allowed.",
                'stage': 'deepfake_detection',
                'nsfw_result': nsfw_result, 'deepfake_result': deepfake_result,
                'user_id': user_id
            })
        
        # Step 3: Pose validation
        pose_result = PoseValidator.validate(temp_path)
        if not pose_result['valid_pose']:
            return jsonify({
                'valid': False, 'reason': "Please upload a clear frontal face photo with proper pose.",
                'stage': 'pose_validation',
                'nsfw_result': nsfw_result, 'deepfake_result': deepfake_result,
                'pose_result': pose_result, 'user_id': user_id
            })
        
        # All validations passed
        return jsonify({
            'valid': True,
            'reason': 'Image passed all validations - suitable for matrimonial profile',
            'stage': 'complete',
            'nsfw_result': nsfw_result,
            'deepfake_result': deepfake_result,
            'pose_result': pose_result,
            'user_id': user_id
        })
        
    except ValidationError as e:
        return jsonify({'valid': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'valid': False, 'error': f'Internal server error: {str(e)}'}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/validate/nsfw', methods=['POST'])
@require_auth
def validate_nsfw_only():
    """NSFW and weapons detection only"""
    temp_path = None
    try:
        temp_path = handle_file_upload()
        result = NSFWValidator.validate(temp_path)
        result['user_id'] = request.user.get('user_id')
        return jsonify(result)
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/validate/deepfake', methods=['POST'])
@require_auth
def validate_deepfake_only():
    """Deepfake detection only"""
    temp_path = None
    try:
        temp_path = handle_file_upload()
        result = DeepfakeValidator.validate(temp_path)
        result['user_id'] = request.user.get('user_id')
        return jsonify(result)
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/validate/pose', methods=['POST'])
@require_auth
def validate_pose_only():
    """Pose validation only"""
    temp_path = None
    try:
        temp_path = handle_file_upload()
        result = PoseValidator.validate(temp_path)
        result['user_id'] = request.user.get('user_id')
        return jsonify(result)
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size: 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ==================== Application Entry Point ====================
if __name__ == '__main__':
    # Load models on startup
    if not model_manager.load_models():
        print("Warning: Could not load all models. API may not function properly.")
        exit(1)
    
    logging.info("API starting with JWT authentication enabled...")
    
    app.run(host='0.0.0.0', port=5000, debug=True)