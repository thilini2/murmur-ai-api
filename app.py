from flask import Flask
from flask_cors import CORS
from config import Config
from models.audio_processor import AudioProcessor
from models.prediction_model import PredictionModel
from services.prediction_service import PredictionService
from api.audio_controller import AudioController, audio_bp
import os

def create_app():
    app = Flask(__name__)
    
    # Enable CORS for the app, allowing requests from http://localhost:3000
    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
    
    # Increase file upload limit (e.g., 16MB)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

    # Create upload folder if it doesn't exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize dependencies
    audio_processor = AudioProcessor(
        Config.SR, Config.SEGMENT_LEN, Config.N_MELS, Config.N_FFT
    )
    prediction_model = PredictionModel(Config.MODEL_PATH)
    prediction_service = PredictionService(
        audio_processor, prediction_model, Config.BEST_THRESHOLD
    )
    
    # Register controller
    AudioController(prediction_service, Config.UPLOAD_FOLDER)
    app.register_blueprint(audio_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)