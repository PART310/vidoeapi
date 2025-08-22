from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
import os
import tempfile

app = Flask(__name__)

# Load model at startup
# model = None
# model_path = '/home/aijohn/mysite/video_model.pkl'

BASE_DIR =os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR,"video_model.pkl")

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    model = None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify(error="Model not loaded"), 500

    # Check file
    if 'file' not in request.files:
        return jsonify(error="No file uploaded"), 400
    file = request.files['file']

    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            file.save(tmp.name)

            # Read video
            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                return jsonify(error="Invalid video file"), 400

            frames = []
            for _ in range(20):  # Take 20 frames
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (64, 64))  # Resize to model input size
                frames.append(frame)

            cap.release()

        if len(frames) == 0:
            return jsonify(error="No frames extracted"), 400

        # Pad if needed
        while len(frames) < 20:
            frames.append(np.zeros((64, 64, 3)))

        # Preprocess
        video_array = np.array(frames) / 255.0
        prediction = model.predict(np.expand_dims(video_array, 0))[0]

        # Extract probabilities
        ai_prob = float(prediction[0])
        real_prob = float(prediction[1])
        result = "AI" if ai_prob > real_prob else "Real"
        confidence = round(max(ai_prob, real_prob) * 100, 2)

        return jsonify(
            result=result,
            confidence=confidence,
            ai_prob=ai_prob,
            real_prob=real_prob
        )

    except Exception as e:
        return jsonify(error=f"Processing failed: {str(e)}"), 500


if __name__ == '__main__':
    app.run(debug=True)
