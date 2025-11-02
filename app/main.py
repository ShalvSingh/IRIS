from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import joblib
import numpy as np
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is one level up from the `app` folder in this repository layout
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
# The model file is stored in the repository-level `model` folder (../model/iris_model.pkl)
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "iris_model.pkl")

try:
    # Provide a clearer error when file is missing
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception:
    # Log full traceback so you can see the exact failure in server logs
    logger.exception(f"Error loading model from {MODEL_PATH}")
    model = None

# Create Flask app with enhanced configuration
app = Flask(__name__)
app.secret_key = 'iris_prediction_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Folders setup
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
IMAGES_FOLDER = os.path.join(BASE_DIR, "static", "images")
LOGS_FOLDER = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, IMAGES_FOLDER, LOGS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["IMAGES_FOLDER"] = IMAGES_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Enhanced flower information with detailed descriptions
FLOWER_INFO = {
    0: {
        "name": "Iris Setosa",
        "emoji": "🌸",
    "image": "setosa.png",
        "description": "Small, delicate petals with vibrant colors. Known for its compact size and early blooming.",
        "characteristics": ["Smallest among iris species", "Petal length: 1.0-1.9 cm", "Distinctive blue-purple color"],
        "confidence_threshold": 0.8
    },
    1: {
        "name": "Iris Versicolor",
        "emoji": "🌿",
    "image": "versicolor.png", 
        "description": "Medium-sized iris with beautiful purple-blue flowers and distinctive markings.",
        "characteristics": ["Medium size", "Petal length: 3.0-5.1 cm", "Purple-blue with yellow markings"],
        "confidence_threshold": 0.75
    },
    2: {
        "name": "Iris Virginica",
        "emoji": "🌺",
    "image": "virginica.png",
        "description": "Largest of the iris species with elegant purple flowers and long petals.",
        "characteristics": ["Largest iris species", "Petal length: 4.5-6.9 cm", "Deep purple with prominent veining"],
        "confidence_threshold": 0.7
    }
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_measurements(sl, sw, pl, pw):
    """Validate measurement inputs with realistic ranges"""
    errors = []
    
    # Check realistic ranges based on iris dataset
    if not (4.0 <= sl <= 8.0):
        errors.append("Sepal length should be between 4.0 and 8.0 cm")
    if not (2.0 <= sw <= 4.5):
        errors.append("Sepal width should be between 2.0 and 4.5 cm") 
    if not (1.0 <= pl <= 7.0):
        errors.append("Petal length should be between 1.0 and 7.0 cm")
    if not (0.1 <= pw <= 2.5):
        errors.append("Petal width should be between 0.1 and 2.5 cm")
    
    return errors

def get_prediction_confidence(features):
    """Get prediction confidence using decision function or probability"""
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([features])[0]
            max_prob = np.max(probabilities)
            return max_prob, probabilities
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function([features])[0]
            # Convert decision scores to confidence-like measure
            confidence = np.max(decision_scores) / (np.max(decision_scores) + np.abs(np.min(decision_scores)))
            return confidence, decision_scores
        else:
            return 0.8, None  # Default confidence
    except Exception as e:
        logger.error(f"Error calculating confidence: {str(e)}")
        return 0.5, None

def log_prediction(input_type, input_data, prediction, confidence):
    """Log prediction for analytics"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_type": input_type,
            "input_data": input_data,
            "prediction": prediction,
            "confidence": confidence,
        }
        
        log_file = os.path.join(LOGS_FOLDER, f"predictions_{datetime.now().strftime('%Y%m')}.json")
        
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")

def process_image(file_path):
    """Enhanced image processing"""
    try:
        with Image.open(file_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Enhance image quality
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)
            
            # Resize if too large
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Save processed image
            img.save(file_path, quality=95, optimize=True)
            
            # Return basic image info for now (placeholder for ML model)
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

@app.route("/")
def home():
    """Homepage route"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Enhanced prediction route with better error handling and features"""
    try:
        if model is None:
            return render_template("index.html", 
                                 prediction="❌ Model not available. Please check server configuration.",
                                 error=True)

        # Case 1: Numeric form inputs
        if "sepal_length" in request.form:
            try:
                # Get and validate inputs
                sl = float(request.form["sepal_length"])
                sw = float(request.form["sepal_width"])
                pl = float(request.form["petal_length"])
                pw = float(request.form["petal_width"])
                
                # Validate measurements
                validation_errors = validate_measurements(sl, sw, pl, pw)
                if validation_errors:
                    return render_template("index.html", 
                                         prediction="❌ " + validation_errors[0],
                                         error=True)
                
                # Prepare features for prediction
                features = [sl, sw, pl, pw]
                
                # Make prediction
                prediction_idx = model.predict([features])[0]
                confidence, probabilities = get_prediction_confidence(features)
                
                # Get flower info
                flower_info = FLOWER_INFO.get(prediction_idx, {})
                flower_name = flower_info.get("name", "Unknown")
                flower_emoji = flower_info.get("emoji", "❓")
                flower_img = flower_info.get("image", "")
                
                # Format prediction result
                prediction_result = f"{flower_emoji} {flower_name} (Confidence: {confidence:.2%})"
                
                # Log the prediction
                log_prediction("numeric", features, prediction_result, confidence)
                
                return render_template("index.html", 
                                     prediction=prediction_result,
                                     flower_img=flower_img)
                
            except ValueError:
                return render_template("index.html", 
                                     prediction="❌ Please enter valid numeric values for all measurements.",
                                     error=True)
        
        # Case 2: Image upload
        elif "flower_image" in request.files:
            file = request.files["flower_image"]
            
            if file.filename == "":
                return render_template("index.html", 
                                     prediction="❌ Please select an image file to upload.",
                                     error=True)
            
            if file and allowed_file(file.filename):
                try:
                    # Secure filename and save
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file.save(filepath)
                    
                    # Process image
                    image_info = process_image(filepath)
                    
                    if image_info:
                        # For now, we'll use a placeholder prediction for images
                        # In a real implementation, you would use a vision model here
                        logger.info(f"Image processed: {image_info}")
                        
                        # Simulate prediction based on image processing
                        # This is a placeholder - in a real app, you'd use a vision model
                        simulated_features = [5.1, 3.5, 1.4, 0.2]  # Default features
                        prediction_idx = model.predict([simulated_features])[0]
                        confidence = 0.85  # Simulated confidence
                        
                        # Get flower info
                        flower_info = FLOWER_INFO.get(prediction_idx, {})
                        flower_name = flower_info.get("name", "Unknown")
                        flower_emoji = flower_info.get("emoji", "❓")
                        flower_img = flower_info.get("image", "")
                        
                        # Format prediction result
                        prediction_result = f"{flower_emoji} {flower_name} (Confidence: {confidence:.2%})"
                        
                        # Log the prediction
                        log_prediction("image", {"filename": filename, "image_info": image_info}, 
                                      prediction_result, confidence)
                        
                        return render_template("index.html", 
                                             prediction=prediction_result,
                                             flower_img=flower_img,
                                             uploaded_image=filename)
                    else:
                        return render_template("index.html", 
                                             prediction="❌ Error processing image. Please try another file.",
                                             error=True)
                        
                except Exception as e:
                    logger.error(f"Error processing image upload: {str(e)}")
                    return render_template("index.html", 
                                         prediction="❌ Error processing image. Please try again.",
                                         error=True)
            else:
                return render_template("index.html", 
                                     prediction="❌ Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, WEBP).",
                                     error=True)
        
        # No valid input provided
        else:
            return render_template("index.html", 
                                 prediction="❌ Please provide either measurements or an image for prediction.",
                                 error=True)
            
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return render_template("index.html", 
                             prediction="❌ An unexpected error occurred. Please try again.",
                             error=True)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return render_template("index.html", 
                         prediction="❌ File too large. Please upload an image smaller than 16MB.",
                         error=True), 413


# Health endpoint for readiness/liveness checks
@app.route("/health")
def health():
    """Return 200 if model is loaded, 503 otherwise. Useful for deployment readiness checks."""
    try:
        if model is None:
            return jsonify(status="unhealthy", reason="model_unavailable"), 503
        # Optionally include model path for debugging (do not expose in production)
        return jsonify(status="ok", model_path=MODEL_PATH), 200
    except Exception:
        logger.exception("Error while evaluating health endpoint")
        return jsonify(status="unhealthy", reason="internal_error"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)