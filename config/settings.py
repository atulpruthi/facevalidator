"""Configuration settings for the matrimonial image validator API."""
import os


class Config:
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'temp_uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'yl_poiuytrewqasdfghjkl')
    JWT_ALGORITHM = 'HS256'
    
    # Model configurations
    NSFW_THRESHOLD = 0.7
    DEEPFAKE_THRESHOLD = 0.7
    WEAPON_THRESHOLD = 0.9
    CELEBRITY_THRESHOLD = 0.7
    FACE_CONFIDENCE_THRESHOLD = 0.7
    PERSON_DETECTION_THRESHOLD = 0.8
    MIN_PERSON_AREA = 5000
    
    # Model paths
    CELEBRITY_MODEL_PATH = './models/celebrity-classifier.pkl'
    
    # Duplicate detection settings
    DUPLICATE_DETECTION_ENABLED = os.getenv('DUPLICATE_DETECTION_ENABLED', 'true').lower() == 'true'
    DUPLICATE_HASH_TYPE = os.getenv('DUPLICATE_HASH_TYPE', 'both')  # 'md5', 'perceptual', or 'both'
    DUPLICATE_SIMILARITY_THRESHOLD = float(os.getenv('DUPLICATE_SIMILARITY_THRESHOLD', '95'))  # Percentage
    DUPLICATE_STORAGE_FILE = os.getenv('DUPLICATE_STORAGE_FILE', 'image_hashes.json')
    DUPLICATE_CLEANUP_DAYS = int(os.getenv('DUPLICATE_CLEANUP_DAYS', '30'))  # Days to keep hashes
    DUPLICATE_CHECK_PER_USER = os.getenv('DUPLICATE_CHECK_PER_USER', 'false').lower() == 'true'  # Check duplicates per user or globally