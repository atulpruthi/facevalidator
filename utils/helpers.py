"""Utility functions for the matrimonial image validator API."""
import os
import io
import base64
import tempfile
import hashlib
import json
from datetime import datetime, timedelta
from PIL import Image
import sys
sys.path.append('..')
from config.settings import Config


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


def generate_image_hash(image_path, hash_type='perceptual'):
    """Generate a hash for the image to detect duplicates
    
    Args:
        image_path (str): Path to the image file
        hash_type (str): Type of hash - 'md5', 'perceptual', or 'both'
    
    Returns:
        dict: Dictionary containing hash(es) and metadata
    """
    try:
        result = {
            'timestamp': datetime.now().isoformat(),
            'filename': os.path.basename(image_path)
        }
        
        # Read image file for MD5 hash
        with open(image_path, 'rb') as f:
            file_content = f.read()
        
        # MD5 hash of file content (exact duplicate detection)
        if hash_type in ['md5', 'both']:
            md5_hash = hashlib.md5(file_content).hexdigest()
            result['md5_hash'] = md5_hash
        
        # Perceptual hash (similar image detection)
        if hash_type in ['perceptual', 'both']:
            try:
                # Open and normalize image
                image = Image.open(image_path)
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize to standard size for consistent hashing
                image = image.resize((64, 64), Image.Resampling.LANCZOS)
                
                # Convert to grayscale for perceptual hashing
                grayscale = image.convert('L')
                
                # Calculate average pixel value
                pixels = list(grayscale.getdata())
                avg_pixel = sum(pixels) / len(pixels)
                
                # Create binary hash based on pixels above/below average
                binary_str = ''.join(['1' if pixel > avg_pixel else '0' for pixel in pixels])
                
                # Convert binary to hex hash
                perceptual_hash = hex(int(binary_str, 2))[2:]
                result['perceptual_hash'] = perceptual_hash
                result['image_size'] = image.size
                
            except Exception as e:
                print(f"Warning: Could not generate perceptual hash: {str(e)}")
                if hash_type == 'perceptual':
                    # Fallback to MD5 if perceptual fails
                    result['md5_hash'] = hashlib.md5(file_content).hexdigest()
        
        return result
        
    except Exception as e:
        raise ValidationError(f"Error generating image hash: {str(e)}")


def calculate_hash_similarity(hash1, hash2):
    """Calculate similarity between two perceptual hashes
    
    Args:
        hash1 (str): First perceptual hash
        hash2 (str): Second perceptual hash
    
    Returns:
        float: Similarity percentage (0-100)
    """
    try:
        # Convert hex hashes to binary
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
        
        # Ensure both binary strings are same length
        max_len = max(len(bin1), len(bin2))
        bin1 = bin1.zfill(max_len)
        bin2 = bin2.zfill(max_len)
        
        # Count matching bits
        matches = sum(b1 == b2 for b1, b2 in zip(bin1, bin2))
        similarity = (matches / max_len) * 100
        
        return similarity
        
    except Exception as e:
        print(f"Error calculating hash similarity: {str(e)}")
        return 0.0


class DuplicateTracker:
    """Simple file-based duplicate image tracker"""
    
    def __init__(self, storage_file='image_hashes.json'):
        self.storage_file = storage_file
        self.hashes = self._load_hashes()
    
    def _load_hashes(self):
        """Load existing hashes from storage file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Warning: Could not load hash storage: {str(e)}")
            return {}
    
    def _save_hashes(self):
        """Save hashes to storage file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.hashes, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save hash storage: {str(e)}")
    
    def check_duplicate(self, image_hash_data, similarity_threshold=95):
        """Check if image is a duplicate
        
        Args:
            image_hash_data (dict): Hash data from generate_image_hash()
            similarity_threshold (float): Similarity threshold for perceptual hashes
        
        Returns:
            dict: Duplicate check result
        """
        try:
            # Check for exact MD5 match
            if 'md5_hash' in image_hash_data:
                md5_hash = image_hash_data['md5_hash']
                for stored_hash in self.hashes.values():
                    if stored_hash.get('md5_hash') == md5_hash:
                        return {
                            'is_duplicate': True,
                            'match_type': 'exact',
                            'similarity': 100.0,
                            'original_upload': stored_hash.get('timestamp'),
                            'original_filename': stored_hash.get('filename')
                        }
            
            # Check for perceptual similarity
            if 'perceptual_hash' in image_hash_data:
                perceptual_hash = image_hash_data['perceptual_hash']
                for stored_hash in self.hashes.values():
                    if 'perceptual_hash' in stored_hash:
                        similarity = calculate_hash_similarity(
                            perceptual_hash, 
                            stored_hash['perceptual_hash']
                        )
                        if similarity >= similarity_threshold:
                            return {
                                'is_duplicate': True,
                                'match_type': 'similar',
                                'similarity': similarity,
                                'original_upload': stored_hash.get('timestamp'),
                                'original_filename': stored_hash.get('filename')
                            }
            
            return {
                'is_duplicate': False,
                'match_type': None,
                'similarity': 0.0
            }
            
        except Exception as e:
            print(f"Error checking duplicate: {str(e)}")
            return {
                'is_duplicate': False,
                'match_type': None,
                'similarity': 0.0,
                'error': str(e)
            }
    
    def add_image(self, image_hash_data, user_id=None):
        """Add image hash to tracker
        
        Args:
            image_hash_data (dict): Hash data from generate_image_hash()
            user_id (str): Optional user ID
        
        Returns:
            str: Unique ID for this image entry
        """
        try:
            # Generate unique ID for this entry
            entry_id = hashlib.md5(
                f"{image_hash_data.get('md5_hash', '')}{image_hash_data.get('timestamp', '')}"
                .encode()
            ).hexdigest()[:16]
            
            # Add user info if provided
            if user_id:
                image_hash_data['user_id'] = user_id
            
            # Store hash data
            self.hashes[entry_id] = image_hash_data
            
            # Save to file
            self._save_hashes()
            
            return entry_id
            
        except Exception as e:
            print(f"Error adding image hash: {str(e)}")
            return None
    
    def cleanup_old_entries(self, days_old=30):
        """Remove hash entries older than specified days
        
        Args:
            days_old (int): Number of days after which to remove entries
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            entries_to_remove = []
            for entry_id, hash_data in self.hashes.items():
                try:
                    entry_date = datetime.fromisoformat(hash_data.get('timestamp', ''))
                    if entry_date < cutoff_date:
                        entries_to_remove.append(entry_id)
                except:
                    # If we can't parse the date, remove the entry
                    entries_to_remove.append(entry_id)
            
            # Remove old entries
            for entry_id in entries_to_remove:
                del self.hashes[entry_id]
            
            if entries_to_remove:
                self._save_hashes()
                print(f"Cleaned up {len(entries_to_remove)} old hash entries")
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


# Global duplicate tracker instance
duplicate_tracker = DuplicateTracker()