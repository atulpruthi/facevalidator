"""Pose validation for detecting proper frontal poses."""
import numpy as np
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not available. Pose validation will be limited.")
from PIL import Image, ImageFilter
import sys
sys.path.append('..')
from config.settings import Config
from utils.helpers import pil_to_base64


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
            if not MEDIAPIPE_AVAILABLE:
                print("MediaPipe not available - skipping person mask creation")
                return None
                
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
            # Lazy import to avoid loading models at import time
            from models.manager import model_manager
            
            # Ensure models are loaded
            if not model_manager.models_loaded:
                if not model_manager.load_models():
                    raise Exception("Failed to load required models")
            
            # Check if detector is available
            if model_manager.face_detector is None:
                raise Exception("Face detector not available")
            
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
            if not MEDIAPIPE_AVAILABLE:
                return {'frontal_face_detected': True, 'face_count': 1, 'reason': 'MediaPipe not available - skipping face detection'}
            
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
            if not MEDIAPIPE_AVAILABLE:
                return {'frontal_pose': True, 'reason': 'MediaPipe not available - skipping pose detection', 'pose_score': 1.0}
                
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