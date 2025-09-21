"""Flask routes for the matrimonial image validator API."""
import os
import logging
from flask import Blueprint, request, jsonify
import sys
sys.path.append('..')
from auth.manager import AuthManager, require_auth
from utils.helpers import (
    ValidationError, allowed_file, process_uploaded_file
)
from validators import NSFWValidator, DeepfakeValidator, PoseValidator, CelebrityValidator
from models.manager import model_manager
from config.settings import Config

logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__)


@api.route('/', methods=['GET'])
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
            'POST /validate/celebrity': 'Celebrity detection',
            'POST /auth/token': 'Generate test token',
            'GET /health': 'Health check'
        },
        'models_loaded': model_manager.models_loaded
    })


@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'face_detector': model_manager.face_detector is not None,
            'nsfw_detector': model_manager.nsfw_detector is not None,
            'deepfake_detector': model_manager.deepfake_detector is not None,
            'weapons_detector': model_manager.weapons_detector is not None,
            'celebrity_classifier': model_manager.celebrity_classifier is not None
        }
    })


@api.route('/auth/token', methods=['POST'])
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
    """Common file upload handling with duplicate detection"""
    file_data = request.files.get('file') or request.files.get('image')
    if not file_data or file_data.filename == '':
        raise ValidationError('No file provided')
    
    if not allowed_file(file_data.filename):
        raise ValidationError('Invalid file type. Allowed: png, jpg, jpeg, gif, bmp')
    
    temp_path = process_uploaded_file(file_data, file_data.filename)
    
    return temp_path


def cleanup_temp_data(temp_path):
    """Clean up temporary file"""
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)


@api.route('/validate', methods=['POST'])
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
                'valid': False, 'reason': 'Please upload image as per instruction given below.',
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
                'valid': False, 'reason': pose_result['reason'],  #"Please upload a clear frontal face photo with proper pose.",
                'stage': 'pose_validation',
                'nsfw_result': nsfw_result, 'deepfake_result': deepfake_result,
                'pose_result': pose_result, 'user_id': user_id
            })
        
        # Step 4: Celebrity detection (optional)
        celebrity_result = CelebrityValidator.detect_celebrity(temp_path)
        print(celebrity_result)
        # Only check for celebrity if the classifier is available and detection succeeded
        if celebrity_result.get('is_celebrity') and model_manager.celebrity_classifier is not None:
            return jsonify({
                'valid': False, 'reason': f"Celebrity detected. Please upload your own photo.",
                'stage': 'celebrity_detection',
                'nsfw_result': nsfw_result, 'deepfake_result': deepfake_result,
                'pose_result': pose_result, 'celebrity_result': celebrity_result,
                'user_id': user_id
            })
        
        # All validations passed
        return jsonify({
            'valid': True,
            'reason': 'Image passed all validations - suitable for matrimonial profile',
            'stage': 'complete',
            'nsfw_result': nsfw_result,
            'deepfake_result': deepfake_result,
            'pose_result': pose_result,
            'celebrity_result': celebrity_result,
            'user_id': user_id
        })
        
    except ValidationError as e:
        return jsonify({'valid': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'valid': False, 'error': f'Internal server error: {str(e)}'}), 500
    finally:
        cleanup_temp_data(temp_path)


@api.route('/validate/nsfw', methods=['POST'])
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
        cleanup_temp_data(temp_path)


@api.route('/validate/deepfake', methods=['POST'])
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
        cleanup_temp_data(temp_path)


@api.route('/validate/pose', methods=['POST'])
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
        cleanup_temp_data(temp_path)


@api.route('/validate/celebrity', methods=['POST'])
@require_auth
def validate_celebrity_only():
    """Celebrity detection only"""
    temp_path = None
    try:
        temp_path = handle_file_upload()
        result = CelebrityValidator.detect_celebrity(temp_path)
        result['user_id'] = request.user.get('user_id')
        
        return jsonify(result)
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    finally:
        cleanup_temp_data(temp_path)


# Error handlers
@api.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size: 16MB'}), 413


@api.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@api.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500