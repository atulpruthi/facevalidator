from PIL import Image, ImageFilter
from networkx import difference
from transformers import pipeline
import mediapipe as mp
import cv2
import numpy as np
import math

def get_largest_box(detections):
    max_area = 0
    largest_box = None
    for det in detections:
        box = det['box']
        width = box['xmax'] - box['xmin']
        height = box['ymax'] - box['ymin']
        area = width * height
        if area > max_area:
            max_area = area
            largest_box = box
    return largest_box

def get_sorted_boxes(detections):
    boxes_with_area = []
    for det in detections:
        box = det['box']
        width = box['xmax'] - box['xmin']
        height = box['ymax'] - box['ymin']
        area = width * height
        boxes_with_area.append((det, area))
    
    boxes_with_area.sort(key=lambda x: x[1], reverse=True)
    return boxes_with_area

def create_person_mask(image_path, box):
    """Create a mask for the person using MediaPipe Selfie Segmentation"""
    try:
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        # Load original image
        original_image = Image.open(image_path)
        original_array = np.array(original_image.convert('RGB'))
        
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
            results = selfie_segmentation.process(original_array)
            
            # Get segmentation mask
            mask = results.segmentation_mask
            
            # Crop the mask to the person's bounding box
            cropped_mask = mask[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
            
            return cropped_mask
    except Exception as e:
        print(f"Error creating person mask: {e}")
        return None

def get_cropped_image_with_blur(image_path, box, blur_radius=15):
    """Crop image and blur background around the person"""
    try:
        # Load original image
        original_image = Image.open(image_path)
        
        # Crop the region
        cropped_image = original_image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
        
        # Create person mask
        person_mask = create_person_mask(image_path, box)
        
        if person_mask is not None:
            # Convert mask to PIL format and resize to match cropped image
            mask_pil = Image.fromarray((person_mask * 255).astype(np.uint8), mode='L')
            mask_pil = mask_pil.resize(cropped_image.size, Image.LANCZOS)
            
            # Create blurred version of cropped image
            blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Composite: use original cropped image for person, blurred for background
            # Invert mask so person area is white (keep original), background is black (use blurred)
            mask_array = np.array(mask_pil)
            
            # Threshold the mask to create cleaner separation
            mask_threshold = 128
            binary_mask = (mask_array > mask_threshold).astype(np.uint8) * 255
            binary_mask_pil = Image.fromarray(binary_mask, mode='L')
            
            # Composite the images
            result_image = Image.composite(cropped_image, blurred_image, binary_mask_pil)
            
            return result_image
        else:
            # Fallback: just crop without blur if segmentation fails
            return cropped_image
            
    except Exception as e:
        print(f"Error in background blur: {e}")
        # Fallback: just crop the image
        image = Image.open(image_path)
        return image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))

def get_cropped_image_simple_blur(image_path, box, blur_radius=10, edge_blur_width=50):
    """Alternative method: blur edges of cropped image"""
    try:
        image = Image.open(image_path)
        cropped = image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
        
        # Create a mask for edge blurring
        width, height = cropped.size
        mask = Image.new('L', (width, height), 255)
        
        # Create gradient mask for edges
        mask_array = np.array(mask)
        
        # Apply gradient blur to edges
        for i in range(edge_blur_width):
            alpha = i / edge_blur_width
            # Top edge
            if i < height:
                mask_array[i, :] = int(255 * alpha)
            # Bottom edge
            if height - 1 - i >= 0:
                mask_array[height - 1 - i, :] = int(255 * alpha)
            # Left edge
            if i < width:
                mask_array[:, i] = np.minimum(mask_array[:, i], int(255 * alpha))
            # Right edge
            if width - 1 - i >= 0:
                mask_array[:, width - 1 - i] = np.minimum(mask_array[:, width - 1 - i], int(255 * alpha))
        
        edge_mask = Image.fromarray(mask_array, mode='L')
        blurred_cropped = cropped.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Composite original center with blurred edges
        result = Image.composite(cropped, blurred_cropped, edge_mask)
        return result
        
    except Exception as e:
        print(f"Error in simple blur: {e}")
        # Fallback
        image = Image.open(image_path)
        return image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))

def validate_image_objects(image_path, use_advanced_blur=True):
    # Load classifier
    classifier = pipeline("object-detection", model="facebook/detr-resnet-50")
    
    # Get detections
    detections = classifier(image_path)
    # Analysis
    print(f"Detections: {detections}")
    persons = [det for det in detections if det['label'] == 'person' and det['score'] > 0.8]

    boxes_with_area = get_sorted_boxes(persons)
    if len(boxes_with_area) < 2:
        if len(boxes_with_area) == 1:
            largest_box, largest_area = boxes_with_area[0]
            if largest_area < 5000:
                return {
                    'reason': "Small image",
                    'has_single_person': False
                }
            else:
                # Choose blur method
                if use_advanced_blur:
                    cropped_image = get_cropped_image_with_blur(image_path, largest_box['box'])
                else:
                    cropped_image = get_cropped_image_simple_blur(image_path, largest_box['box'])
                
                return {
                    'image': cropped_image,
                    'reason': "Valid single person",
                    'has_single_person': True,
                }
        else:
            return {
                'reason': "No persons detected",
                'has_single_person': False,
            }
    else:
        largest_box, largest_area = boxes_with_area[0]
        second_largest_box, second_largest_area = boxes_with_area[1]
        difference = largest_area - second_largest_area
        percentage_diff = (difference / second_largest_area) * 100 if second_largest_area > 0 else float('inf')
        if percentage_diff < 200:
            return {
                'reason': "Multiple persons detected",
                'has_single_person': False,
            }
        else:
            # Choose blur method
            if use_advanced_blur:
                cropped_image = get_cropped_image_with_blur(image_path, largest_box['box'])
            else:
                cropped_image = get_cropped_image_simple_blur(image_path, largest_box['box'])
            
            return {
                'image': cropped_image,
                'reason': "Valid single person",
                'has_single_person': True,
            }

def detect_frontal_face_mediapipe(pil_image):
    """Detect frontal faces using MediaPipe"""
    try:
        mp_face_detection = mp.solutions.face_detection
        
        # Convert PIL Image directly to RGB numpy array
        rgb_array = np.array(pil_image.convert('RGB'))
        
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb_array)
            
            if not results.detections:
                return {
                    'frontal_face_detected': False,
                    'face_count': 0,
                    'reason': 'No faces detected'
                }
            
            if len(results.detections) > 1:
                return {
                    'frontal_face_detected': False,
                    'face_count': len(results.detections),
                    'reason': 'Multiple faces detected'
                }
            
            # Get detection confidence
            detection = results.detections[0]
            confidence = detection.score[0]
            
            # Check if confidence is high enough for frontal face
            is_frontal = confidence > 0.7
            
            return {
                'frontal_face_detected': is_frontal,
                'face_count': 1,
                'confidence': confidence,
                'reason': f'Frontal face detected with confidence {confidence:.2f}' if is_frontal else 'Low confidence frontal face'
            }
            
    except Exception as e:
        return {
            'frontal_face_detected': False,
            'face_count': 0,
            'reason': f'Error in face detection: {str(e)}'
        }

def calculate_angle(point1, point2):
    """Calculate angle between two points"""
    return math.atan2(point2[1] - point1[1], point2[0] - point1[0]) * 180 / math.pi

def detect_pose_orientation(pil_image):
    """Detect if person is in side pose using MediaPipe Pose with eye visibility"""
    try:
        mp_pose = mp.solutions.pose
        
        # Convert PIL Image to RGB numpy array
        rgb_array = np.array(pil_image.convert('RGB'))
        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose:
            
            results = pose.process(rgb_array)
            
            if not results.pose_landmarks:
                return {
                    'frontal_pose': False,
                    'reason': 'No pose landmarks detected',
                    'pose_score': 0.0
                }
            
            landmarks = results.pose_landmarks.landmark
            h, w = rgb_array.shape[:2]
            
            # Key pose landmarks for side pose detection
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
            left_eye_inner = landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value]
            right_eye_inner = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value]
            left_eye_outer = landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value]
            right_eye_outer = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value]
            
            # Convert to pixel coordinates
            left_shoulder_px = (left_shoulder.x * w, left_shoulder.y * h)
            right_shoulder_px = (right_shoulder.x * w, right_shoulder.y * h)
            nose_px = (nose.x * w, nose.y * h)
            left_ear_px = (left_ear.x * w, left_ear.y * h)
            right_ear_px = (right_ear.x * w, right_ear.y * h)
            left_eye_px = (left_eye.x * w, left_eye.y * h)
            right_eye_px = (right_eye.x * w, right_eye.y * h)
            
            # Check visibility for all key landmarks
            left_shoulder_visible = left_shoulder.visibility > 0.5
            right_shoulder_visible = right_shoulder.visibility > 0.5
            left_ear_visible = left_ear.visibility > 0.5
            right_ear_visible = right_ear.visibility > 0.5
            left_eye_visible = left_eye.visibility > 0.5
            right_eye_visible = right_eye.visibility > 0.5
            left_eye_inner_visible = left_eye_inner.visibility > 0.5
            right_eye_inner_visible = right_eye_inner.visibility > 0.5
            left_eye_outer_visible = left_eye_outer.visibility > 0.5
            right_eye_outer_visible = right_eye_outer.visibility > 0.5
            
            # Calculate shoulder width ratio
            shoulder_distance = abs(left_shoulder_px[0] - right_shoulder_px[0])
            image_width = w
            shoulder_width_ratio = shoulder_distance / image_width
            
            # Calculate ear visibility ratio
            ear_visibility_ratio = (left_ear.visibility + right_ear.visibility) / 2
            
            # Calculate eye visibility metrics
            eye_visibility_ratio = (left_eye.visibility + right_eye.visibility) / 2
            eye_inner_visibility_ratio = (left_eye_inner.visibility + right_eye_inner.visibility) / 2
            eye_outer_visibility_ratio = (left_eye_outer.visibility + right_eye_outer.visibility) / 2
            
            # Calculate eye symmetry (important for frontal pose)
            eye_visibility_balance = abs(left_eye.visibility - right_eye.visibility)
            both_eyes_visible = left_eye_visible and right_eye_visible
            
            # Calculate eye distance ratio (eyes should be appropriately spaced for frontal pose)
            if both_eyes_visible:
                eye_distance = abs(left_eye_px[0] - right_eye_px[0])
                eye_distance_ratio = eye_distance / image_width
            else:
                eye_distance_ratio = 0.0
            
            # Calculate nose position relative to shoulders
            if left_shoulder_visible and right_shoulder_visible:
                shoulder_center_x = (left_shoulder_px[0] + right_shoulder_px[0]) / 2
                nose_offset = abs(nose_px[0] - shoulder_center_x) / image_width
            else:
                nose_offset = 1.0  # Assume side pose if shoulders not visible
            
            # Calculate nose position relative to eyes
            if both_eyes_visible:
                eye_center_x = (left_eye_px[0] + right_eye_px[0]) / 2
                nose_eye_offset = abs(nose_px[0] - eye_center_x) / image_width
            else:
                nose_eye_offset = 1.0
            
            # Enhanced frontal pose detection criteria
            is_frontal = (
                left_shoulder_visible and right_shoulder_visible and  # Both shoulders visible
                both_eyes_visible and  # Both eyes visible
                shoulder_width_ratio > 0.15 and  # Sufficient shoulder width
                eye_distance_ratio > 0.08 and  # Sufficient eye separation for frontal pose
                eye_distance_ratio < 0.35 and  # Not too wide (could be distortion)
                nose_offset < 0.1 and  # Nose centered relative to shoulders
                nose_eye_offset < 0.08 and  # Nose centered relative to eyes
                ear_visibility_ratio > 0.3 and  # Both ears somewhat visible
                eye_visibility_ratio > 0.6 and  # Both eyes clearly visible
                eye_visibility_balance < 0.4 and  # Eye visibility balanced
                abs(left_ear.visibility - right_ear.visibility) < 0.5  # Ear visibility balance
            )
            
            # Calculate comprehensive pose score
            pose_score = min(1.0, (
                shoulder_width_ratio * 1.5 + 
                eye_distance_ratio * 3 +
                (1 - nose_offset) * 2 + 
                (1 - nose_eye_offset) * 2 +
                ear_visibility_ratio * 1.5 +
                eye_visibility_ratio * 2 +
                (1 - eye_visibility_balance) * 1.5 +
                (1 - abs(left_ear.visibility - right_ear.visibility)) * 0.5
            ) / 12)
            
            # Detailed rejection reasons
            if not is_frontal:
                if not (left_shoulder_visible and right_shoulder_visible):
                    reason = "Side pose detected - one shoulder not visible"
                elif not both_eyes_visible:
                    reason = f"Side pose detected - one eye not visible (L:{left_eye.visibility:.2f}, R:{right_eye.visibility:.2f})"
                elif shoulder_width_ratio <= 0.15:
                    reason = f"Side pose detected - narrow shoulder width ({shoulder_width_ratio:.2f})"
                elif eye_distance_ratio <= 0.08:
                    reason = f"Side pose detected - eyes too close ({eye_distance_ratio:.2f})"
                elif eye_distance_ratio >= 0.35:
                    reason = f"Side pose detected - eyes too wide ({eye_distance_ratio:.2f})"
                elif nose_offset >= 0.1:
                    reason = f"Side pose detected - nose not centered to shoulders (offset: {nose_offset:.2f})"
                elif nose_eye_offset >= 0.08:
                    reason = f"Side pose detected - nose not centered to eyes (offset: {nose_eye_offset:.2f})"
                elif ear_visibility_ratio <= 0.3:
                    reason = f"Side pose detected - low ear visibility ({ear_visibility_ratio:.2f})"
                elif eye_visibility_ratio <= 0.6:
                    reason = f"Side pose detected - low eye visibility ({eye_visibility_ratio:.2f})"
                elif eye_visibility_balance >= 0.4:
                    reason = f"Side pose detected - unbalanced eye visibility ({eye_visibility_balance:.2f})"
                else:
                    reason = "Side pose detected - asymmetric pose"
            else:
                reason = f"Frontal pose detected (score: {pose_score:.2f})"
            
            return {
                'frontal_pose': is_frontal,
                'reason': reason,
                'pose_score': pose_score,
                'shoulder_width_ratio': shoulder_width_ratio,
                'eye_distance_ratio': eye_distance_ratio,
                'nose_offset': nose_offset,
                'nose_eye_offset': nose_eye_offset,
                'ear_visibility_ratio': ear_visibility_ratio,
                'eye_visibility_ratio': eye_visibility_ratio,
                'eye_visibility_balance': eye_visibility_balance,
                'shoulders_visible': left_shoulder_visible and right_shoulder_visible,
                'both_eyes_visible': both_eyes_visible,
                'eye_details': {
                    'left_eye_visible': left_eye_visible,
                    'right_eye_visible': right_eye_visible,
                    'left_eye_visibility': left_eye.visibility,
                    'right_eye_visibility': right_eye.visibility,
                    'left_eye_inner_visible': left_eye_inner_visible,
                    'right_eye_inner_visible': right_eye_inner_visible,
                    'left_eye_outer_visible': left_eye_outer_visible,
                    'right_eye_outer_visible': right_eye_outer_visible
                }
            }
            
    except Exception as e:
        return {
            'frontal_pose': False,
            'reason': f'Error in pose detection: {str(e)}',
            'pose_score': 0.0
        }

def validate_matrimonial_image_enhanced(image_path, use_advanced_blur=True):
    # Step 1: Object detection for person with background blur
    object_result = validate_image_objects(image_path, use_advanced_blur=use_advanced_blur)
    
    if not object_result['has_single_person']:
        return object_result
    
    img = object_result.get('image')
    if img is None:
        return {
            'reason': 'No valid image for analysis',
            'has_single_person': False,
            'frontal_face': False,
            'frontal_pose': False
        }
    
    # Step 2: Face detection
    face_result = detect_frontal_face_mediapipe(img)
    
    if not face_result['frontal_face_detected']:
        return {
            'reason': f"Face issue: {face_result['reason']}",
            'has_single_person': True,
            'frontal_face': False,
            'frontal_pose': False
        }
    
    # Step 3: Pose orientation detection with eye visibility
    pose_result = detect_pose_orientation(img)
    
    if not pose_result['frontal_pose']:
        return {
            'reason': f"Pose issue: {pose_result['reason']}",
            'has_single_person': True,
            'frontal_face': True,
            'frontal_pose': False,
            'face_confidence': face_result['confidence'],
            'pose_score': pose_result['pose_score'],
            'eye_details': pose_result.get('eye_details', {}),
            'processed_image': img  # Return the blurred background image
        }
    
    # All checks passed
    return {
        'reason': 'Valid matrimonial photo - frontal face and pose with visible eyes',
        'has_single_person': True,
        'frontal_face': True,
        'frontal_pose': True,
        'face_confidence': face_result['confidence'],
        'pose_score': pose_result['pose_score'],
        'eye_visibility_ratio': pose_result['eye_visibility_ratio'],
        'eye_details': pose_result['eye_details'],
        'processed_image': img  # Return the blurred background image
    }

# Usage
result = validate_matrimonial_image_enhanced(r"C:\Users\91965\Downloads\3.jpg", use_advanced_blur=True)
print(f"Final result: {result}")

# Save the processed image if validation passed
if result.get('processed_image'):
    result['processed_image'].save('processed_matrimonial_photo.jpg', 'JPEG', quality=95)
    print("Processed image saved as 'processed_matrimonial_photo.jpg'")