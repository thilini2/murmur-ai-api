# api/audio_controller.py
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from config import Config

audio_bp = Blueprint('audio', __name__)

class AudioController:
    def __init__(self, prediction_service, upload_folder):
        self.prediction_service = prediction_service
        self.upload_folder = upload_folder
        self._register_routes()

    def _register_routes(self):
        audio_bp.add_url_rule('/predict/single', 'predict_single', self.predict_single, methods=['POST'])
        audio_bp.add_url_rule('/predict/batch', 'predict_batch', self.predict_batch, methods=['POST'])

    def predict_single(self):
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(self.upload_folder, filename)
        file.save(filepath)
        
        try:
            result = self.prediction_service.predict_single(filepath)
            os.remove(filepath)
            return jsonify(result)
        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": str(e)}), 500

    def predict_batch(self):
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
            
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files selected"}), 400
        
        print(f"Received {len(files)} files for processing")
        
        filepaths = []
        try:
            for i, file in enumerate(files):
                if file.filename == '':
                    print(f"Skipping file {i+1}: Empty filename")
                    continue
                filename = secure_filename(file.filename)
                filepath = os.path.join(self.upload_folder, filename)
                file.save(filepath)
                filepaths.append(filepath)
                print(f"Saved file {i+1}: {filepath}")
                
            print(f"Total files to process: {len(filepaths)}")
            result = self.prediction_service.predict_batch(filepaths, labels_df=None)
            
            return jsonify({
                "prediction": result["prediction"],
                "results": result["results"],
                "batch_summary": result["batch_summary"],  
                "predictions_avg": result["predictions_avg"],
                "probabilities_avg": result["probabilities_avg"]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            for filepath in filepaths:
                if os.path.exists(filepath):
                    os.remove(filepath)