
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
import pickle
import numpy as np

app = Flask(__name__)

scaler = joblib.load('models/exoplanet_scaler.pkl')
label_encoder = joblib.load('models/exoplanet_label_encoder.pkl')

with open('models/model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.layer_1 = nn.Linear(input_size, 128)
        self.layer_4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_4(x)
        return x

model = Classifier(model_info['input_size'], model_info['num_classes'])
model.load_state_dict(torch.load('models/exoplanet_classifier_model.pth', weights_only=True))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']

        if len(features) != 30:
            return jsonify({'error': f'Expected 30 features, got {len(features)}'}), 400

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        prediction = predicted.item()
        confidence = probabilities[0][prediction].item()
        label = label_encoder.inverse_transform([prediction])[0]

        return jsonify({
            'prediction': int(prediction),
            'confidence': float(confidence),
            'label': label,
            'probabilities': {
                'CANDIDATE': float(probabilities[0][0]),
                'CONFIRMED': float(probabilities[0][1])
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({'feature_names': model_info['feature_names']})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
