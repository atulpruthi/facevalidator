"""Authentication module for JWT token management."""
import jwt
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
import sys
sys.path.append('..')
from config.settings import Config

logger = logging.getLogger(__name__)


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