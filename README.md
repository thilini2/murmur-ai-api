
# Audio Classifier API

This is a Flask-based REST API for classifying audio files (.wav) as "Normal" or "Abnormal" using a pre-trained TensorFlow model. The API processes audio files by converting them to Mel-spectrograms and provides predictions based on both average and maximum probability methods.

## Features
- Single audio file prediction endpoint
- Batch audio file prediction endpoint
- Detailed output including segment probabilities and batch statistics
- Clean architecture following SOLID principles

## Prerequisites
- Python 3.8+
- A pre-trained model file (`final_model.keras`) placed in the project root
- WAV audio files for testing

## Project Structure
```
audio_classifier_api/
├── app.py              # Main Flask application
├── config.py          # Configuration settings
├── models/            # Audio processing and model handling
├── services/          # Business logic
├── api/               # API endpoints
├── requirements.txt   # Project dependencies
└── uploads/           # Temporary folder for uploaded files (created at runtime)
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd audio_classifier_api
```

### 2. Create a Virtual Environment
Create an isolated Python environment to manage dependencies:
```bash
python3 -m venv env
```

### 3. Activate the Virtual Environment
Activate the virtual environment based on your operating system:

- **Windows:**
  ```bash
  env\Scriptsctivate
  ```

- **macOS/Linux:**
  ```bash
  source env/bin/activate
  ```

You should see `(env)` in your terminal prompt indicating the virtual environment is active.

### 4. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 5. Verify Dependencies
Check installed packages:
```bash
pip list
```
Expected packages include Flask, NumPy, TensorFlow, Librosa, and Werkzeug.

### 6. Prepare the Model
Ensure `final_model.keras` is in the project root directory. If it's located elsewhere, update the `MODEL_PATH` in `config.py`.

### 7. Run the Application
Start the Flask development server:
```bash
python app.py
```
The API will be available at `http://localhost:5000`.

### 8. Deactivate the Virtual Environment
When finished, deactivate the virtual environment:
```bash
deactivate
```

## API Endpoints

### Single Prediction
- **Endpoint:** `POST /predict/single`
- **Description:** Predict classification for a single WAV file
- **Request:**
  ```bash
  curl -X POST -F "file=@path/to/audio.wav" http://localhost:5000/predict/single
  ```
- **Response:**
  ```json
  {
    "label": "Normal" | "Abnormal",
    "probability": 45.67,
    "segment_probabilities": [12.34, 23.45, ...]
  }
  ```

### Batch Prediction
- **Endpoint:** `POST /predict/batch`
- **Description:** Predict classifications for multiple WAV files
- **Request:**
  ```bash
  curl -X POST -F "files=@audio1.wav" -F "files=@audio2.wav" http://localhost:5000/predict/batch
  ```
- **Response:**
  ```json
  {
    "results": [
      {
        "patient_id": "audio1",
        "avg_probability": 0.4567,
        "max_probability": 0.7890,
        "prediction_avg": "Normal",
        "prediction_max": "Abnormal",
        "segment_probabilities": [0.1234, 0.2345, ...]
      },
      ...
    ],
    "batch_summary": "Batch Testing Results...",
    "predictions_avg": [0, 1, ...],
    "probabilities_avg": [0.4567, 0.6789, ...]
  }
  ```

## Console Output
The batch prediction endpoint prints detailed analysis to the console, including:
- Per-file predictions with average and max probabilities
- Segment-by-segment probabilities
- Batch summary statistics

## Troubleshooting
- **Missing Model File:** Ensure `final_model.keras` exists in the root directory.
- **Dependency Issues:** Verify all packages are installed correctly in the virtual environment.
- **File Upload Errors:** Check file permissions and ensure WAV files are valid.

## Development Notes
- The API follows SOLID principles and clean code practices
- Temporary uploaded files are automatically cleaned up
- Error handling is implemented for common scenarios

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your changes

## License
[Specify your license here, e.g., MIT]
