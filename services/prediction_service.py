# services/prediction_service.py
from models.audio_processor import AudioProcessor
from models.prediction_model import PredictionModel
import numpy as np

class PredictionService:
    def __init__(self, audio_processor: AudioProcessor, prediction_model: PredictionModel, threshold: float):
        self.audio_processor = audio_processor
        self.prediction_model = prediction_model
        self.threshold = threshold

    def predict_single(self, wav_path):
        input_data = self.audio_processor.wav_to_mel_spectrograms(wav_path)
        probabilities = self._predict_segments(input_data)
        avg_prob = sum(probabilities) / len(probabilities)
        label = "Abnormal" if avg_prob > self.threshold else "Normal"
        return {
            "label": label,
            "probability": float(avg_prob * 100),
            "segment_probabilities": [float(p * 100) for p in probabilities]
        }

    def predict_batch(self, wav_paths, labels_df=None):
        print("\nStarting batch testing...")
        
        results = []
        predictions_avg = []
        predictions_max = []
        predictions_weighted = []  
        probabilities_avg = []
        probabilities_weighted = []  
        all_segment_probs = []
        
        for wav_path in wav_paths:
            patient_id = wav_path.split('/')[-1].replace('.wav', '')
            print(f"\nProcessing {wav_path} (Patient ID: {patient_id})...")
            
            input_data = self.audio_processor.wav_to_mel_spectrograms(wav_path)
            segment_probs = self._predict_segments(input_data)
            
            # Average probability method
            prob_avg = np.mean(segment_probs)
            pred_avg = int(prob_avg > self.threshold)
            class_label_avg = "Abnormal" if pred_avg else "Normal"
            
            # Max probability method
            prob_max = np.max(segment_probs)
            pred_max = int(prob_max > self.threshold)
            class_label_max = "Abnormal" if pred_max else "Normal"
            
            # Weighted average method (50/50 weighting of prob_avg and prob_max)
            w1 = 0.5  # Weight for average probability
            w2 = 0.5  # Weight for max probability
            prob_weighted = (w1 * prob_avg + w2 * prob_max) / (w1 + w2)
            pred_weighted = int(prob_weighted > self.threshold)
            class_label_weighted = "Abnormal" if pred_weighted else "Normal"
            
            print(f"Average probability: {prob_avg:.4f} -> Prediction (Avg): {class_label_avg}")
            print(f"Max probability: {prob_max:.4f} -> Prediction (Max): {class_label_max}")
            print(f"Weighted average probability: {prob_weighted:.4f} -> Prediction (Weighted Avg): {class_label_weighted}")
            print(f"Segment probabilities: {[f'{p:.4f}' for p in segment_probs]}")
            
            predictions_avg.append(pred_avg)
            predictions_max.append(pred_max)
            predictions_weighted.append(pred_weighted)  # Add weighted prediction
            probabilities_avg.append(float(prob_avg))
            probabilities_weighted.append(float(prob_weighted))  # Add weighted probability
            all_segment_probs.append(segment_probs)
            
            results.append({
                "patient_id": patient_id,
                "avg_probability": float(prob_avg),
                "max_probability": float(prob_max),
                "weighted_probability": float(prob_weighted),  # Add weighted probability to results
                "prediction_avg": class_label_avg,
                "prediction_max": class_label_max,
                "prediction_weighted": class_label_weighted,  # Add weighted prediction to results
                "segment_probabilities": [float(p) for p in segment_probs]
            })

        # Calculate batch statistics
        total_files = len(wav_paths)
        response = self._format_batch_response(
            results, predictions_avg, predictions_max, predictions_weighted, probabilities_avg, probabilities_weighted, total_files
        )
        
        # Handle labels if provided
        if labels_df is not None:
            response.update(self._calculate_accuracy(wav_paths, labels_df, predictions_avg, predictions_max, probabilities_avg))
        
        return response

    def _predict_segments(self, input_data):
        probabilities = []
        for i in range(10):
            segment_data = input_data[i]
            prob = self.prediction_model.predict(segment_data)[0, 0]
            probabilities.append(float(prob))
        return probabilities

    def _format_batch_response(self, results, predictions_avg, predictions_max, predictions_weighted, probabilities_avg, probabilities_weighted, total_files):
        num_normal_avg = predictions_avg.count(0)
        num_abnormal_avg = predictions_avg.count(1)
        normal_ratio_avg = (num_normal_avg / total_files) * 100
        abnormal_ratio_avg = (num_abnormal_avg / total_files) * 100

        num_normal_max = predictions_max.count(0)
        num_abnormal_max = predictions_max.count(1)
        normal_ratio_max = (num_normal_max / total_files) * 100
        abnormal_ratio_max = (num_abnormal_max / total_files) * 100

        num_normal_weighted = predictions_weighted.count(0)
        num_abnormal_weighted = predictions_weighted.count(1)
        normal_ratio_weighted = (num_normal_weighted / total_files) * 100
        abnormal_ratio_weighted = (num_abnormal_weighted / total_files) * 100

        # Calc the final result (using weighted average predictions)
        final = ''
        if num_normal_weighted > num_abnormal_weighted:
            final = 'Normal'
        else:
            final = 'Abnormal'

        # Console output (updated to include weighted average method)
        console_output = "Batch Testing Results (Average Probability Method):\n"
        console_output += f"Total files tested: {total_files}\n"
        console_output += f"Normal predictions: {num_normal_avg} ({normal_ratio_avg:.2f}%)\n"
        console_output += f"Abnormal predictions: {num_abnormal_avg} ({abnormal_ratio_avg:.2f}%)\n"
        console_output += "\nBatch Testing Results (Max Probability Method):\n"
        console_output += f"Total files tested: {total_files}\n"
        console_output += f"Normal predictions: {num_normal_max} ({normal_ratio_max:.2f}%)\n"
        console_output += f"Abnormal predictions: {num_abnormal_max} ({abnormal_ratio_max:.2f}%)\n"
        console_output += "\nBatch Testing Results (Weighted Average Method):\n"
        console_output += f"Total files tested: {total_files}\n"
        console_output += f"Normal predictions: {num_normal_weighted} ({normal_ratio_weighted:.2f}%)\n"
        console_output += f"Abnormal predictions: {num_abnormal_weighted} ({abnormal_ratio_weighted:.2f}%)\n"
        console_output += f"Prediction: {final}\n"

        print(console_output)

        # Format batch_summary as a list of objects for JSON response
        batch_summary = [
            {
                "method": "Average Probability",
                "totalFilesTested": total_files,
                "normalPredictions": f"{num_normal_avg} {normal_ratio_avg:.2f}",
                "abnormalPredictions": f"{num_abnormal_avg} {abnormal_ratio_avg:.2f}"
            },
            {
                "method": "Max Probability",
                "totalFilesTested": total_files,
                "normalPredictions": f"{num_normal_max} {normal_ratio_max:.2f}",
                "abnormalPredictions": f"{num_abnormal_max} {abnormal_ratio_max:.2f}"
            }
        ]

        return {
            "prediction": final,
            "results": results,
            "batch_summary": batch_summary,  # Updated to include weighted average
            "predictions_avg": predictions_avg,
            "probabilities_avg": [float(p) for p in probabilities_avg]
        }

    def _calculate_accuracy(self, wav_paths, labels_df, predictions_avg, predictions_max, probabilities_avg):
        true_labels = []
        for wav_path in wav_paths:
            patient_id = wav_path.split('/')[-1].replace('.wav', '')
            label_row = labels_df[labels_df['patient_id'] == patient_id]
            true_labels.append(label_row['label'].values[0] if not label_row.empty else None)

        valid_indices = [i for i, tl in enumerate(true_labels) if tl is not None]
        valid_true_labels = [true_labels[i] for i in valid_indices]
        valid_pred_avg = [predictions_avg[i] for i in valid_indices]
        valid_pred_max = [predictions_max[i] for i in valid_indices]
        valid_probs_avg = [probabilities_avg[i] for i in valid_indices]

        accuracy_output = ""
        if valid_true_labels:
            # Average method accuracy
            correct_avg = sum(1 for true, pred in zip(valid_true_labels, valid_pred_avg) if true == pred)
            accuracy_avg = (correct_avg / len(valid_true_labels)) * 100
            accuracy_output += f"\nAccuracy (Average Probability Method): {correct_avg}/{len(valid_true_labels)} correct ({accuracy_avg:.2f}%)\n"
            
            # Max method accuracy
            correct_max = sum(1 for true, pred in zip(valid_true_labels, valid_pred_max) if true == pred)
            accuracy_max = (correct_max / len(valid_true_labels)) * 100
            accuracy_output += f"Accuracy (Max Probability Method): {correct_max}/{len(valid_true_labels)} correct ({accuracy_max:.2f}%)\n"
            
            # Threshold testing
            accuracy_output += "\nTesting different thresholds for Average Probability Method:\n"
            thresholds = np.arange(0.2, 0.5, 0.05)
            for thresh in thresholds:
                pred_labels_thresh = [int(prob > thresh) for prob in valid_probs_avg]
                correct = sum(1 for true, pred in zip(valid_true_labels, pred_labels_thresh) if true == pred)
                accuracy = (correct / len(valid_true_labels)) * 100
                accuracy_output += f"Threshold {thresh:.2f}: {correct}/{len(valid_true_labels)} correct ({accuracy:.2f}%)\n"
            
            print(accuracy_output)

        return {"accuracy_results": accuracy_output}