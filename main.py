"""Main application file for the matrimonial image validator API.

This is the refactored version of combined_validator_api.py, now organized into modular components.
"""
import os
import logging
from flask import Flask
import sys
sys.path.append('.')
from config.settings import Config
from models.manager import model_manager
from routes import api

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configure Flask app
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.secret_key = Config.JWT_SECRET_KEY

# Create upload folder if it doesn't exist
if not os.path.exists(Config.UPLOAD_FOLDER):
    os.makedirs(Config.UPLOAD_FOLDER)

# Register blueprints
app.register_blueprint(api)

# Global error handlers for the entire app
@app.errorhandler(413)
def too_large(e):
    from flask import jsonify
    return jsonify({'error': 'File too large. Maximum size: 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    from flask import jsonify
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    from flask import jsonify
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Main entry point for the application."""
    # Load models on startup
    if not model_manager.load_models():
        print("Warning: Could not load all models. API may not function properly.")
        exit(1)
    
    logging.info("API starting with JWT authentication enabled...")
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()