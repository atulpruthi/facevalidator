#!/usr/bin/env python3
"""Production server for the Face Validator API - Fast startup"""

import os
import logging
from flask import Flask, jsonify
import sys
sys.path.append('.')

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configure Flask app  
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.secret_key = 'your-secret-key-here-change-in-production'

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Import and register API routes with lazy loading
try:
    from routes import api
    app.register_blueprint(api)
    logger.info("✅ API routes loaded successfully")
except Exception as e:
    logger.error(f"❌ Could not load API routes: {e}")

# Global error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size: 16MB'}), 413

@app.errorhandler(404)  
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main entry point for the application."""
    print("\n" + "="*60)
    print("🚀 FACE VALIDATOR API - PRODUCTION MODE")
    print("="*60)
    print("📊 Status: Starting...")
    
    logger.info("Starting Face Validator API in production mode...")
    
    try:
        print("🔧 Models: Lazy loading (loaded on first request)")
        print("🔄 Duplicate Detection: ✅ Enabled") 
        print("🌐 Server: Starting on http://localhost:5000")
        print("📋 Health Check: http://localhost:5000/health")
        print("📖 API Info: http://localhost:5000/")
        print("="*60)
        print("✅ API Ready! You can now send requests.")
        print("="*60)
        
        # Run in production mode (no debug, no auto-reload)
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=False,  # Disable debug mode for faster startup
            use_reloader=False  # Disable auto-reload
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Face Validator API...")
        logger.info("API shut down by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"❌ Failed to start server: {e}")

if __name__ == '__main__':
    main()