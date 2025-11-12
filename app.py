# app.py (Consolidated Version with Gemini AI + History + Calculator + MONGODB ATLAS)

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import logging
import uuid
from datetime import datetime
import shutil
import json
import google.generativeai as genai

# --- NEW: MongoDB Imports ---
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    
# --- Logging (Moved up to be used by Config) ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    # --- UPDATED: MongoDB Atlas Configuration ---
    MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")
    MONGO_DB_NAME = "kuih_db"
    
    if not MONGO_DB_PASSWORD:
        logging.warning("MONGO_DB_PASSWORD environment variable not set. Database connection will fail.")
        MONGO_URI = None # Set to None to cause a graceful failure
    else:
        # Build the full URI from your template
        MONGO_URI = f"mongodb+srv://kuihdb:{MONGO_DB_PASSWORD}@kuihdb.rcqmsst.mongodb.net/?appName=kuihdb"

    # --- Other Config (Unchanged) ---
    UPLOAD_FOLDER = 'uploads'
    FEEDBACK_FOLDER = 'feedback_images'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    MODEL_PATH = 'kuih_recognition_model.keras'
    TARGET_SIZE = (224, 224)
    MIN_CONFIDENCE_THRESHOLD = 0.7

# --- Flask App Setup ---
app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.FEEDBACK_FOLDER, exist_ok=True)

# --- Global Variables ---
model = None
class_labels = ['Akok', 'Cek Mek Molek','Ketayap', 'Kole Kacang', 'Kuih Bakar', 'Kuih Lapis', 'Kuih Lompang', 'Kuih Qasidah', 'Onde-onde', 'Pulut Sekaya', 'Seri Muka']
metrics = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0}
model_loaded = False

# --- NEW: MongoDB Globals ---
client = None
db = None
db_connection_ok = False

# --- NEW: Gemini AI Configuration (Unchanged) ---
try:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        logger.warning("GEMINI_API_KEY environment variable not set. Gemini features will be disabled.")
        GEMINI_AVAILABLE = False
    else:
        genai.configure(api_key=API_KEY)
        GEMINI_AVAILABLE = True
except Exception as e:
    logger.error(f"Error configuring Gemini AI: {e}")
    GEMINI_AVAILABLE = False

GEMINI_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
    "estimatedcalories":{
            "type": "STRING",
            "description": "A short brief, estimated the calories based on the standard size (g), piece and sources."
        },
        "othersname": {
            "type": "STRING",
            "description": "Other names or local variations of the kuih."
        },
        "description": {
            "type": "STRING",
            "description": "A brief, engaging 2-3 sentence description of the kuih."
        },
        "fun_fact": {
            "type": "STRING",
            "description": "A single interesting fun fact about the kuih's history, ingredients, or cultural significance"
        }
    },
    "required": ["estimatedcalories","othersname","description", "fun_fact"]
}

# --- UPDATED: Database Helpers (MongoDB Atlas) ---
def init_db():
    """Initializes the MongoDB client and database object."""
    global client, db, db_connection_ok
    if not MONGO_AVAILABLE:
        logger.warning("Pymongo not found. MongoDB features disabled.")
        return

    # Get the fully constructed URI from the app config
    mongo_uri = app.config.get("MONGO_URI")

    if not mongo_uri:
        logger.error("MONGO_URI not configured. Did you set the MONGO_DB_PASSWORD environment variable?")
        db_connection_ok = False
        return

    try:
        # Create a single client. serverSelectionTimeoutMS checks connection within 5 secs.
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info() # Force connection test
        db = client[Config.MONGO_DB_NAME]
        db_connection_ok = True
        logger.info("MongoDB Atlas connection successful.")
    except ConnectionFailure as e:
        logger.error(f"MongoDB Atlas connection FAILED: {e}")
        db_connection_ok = False
        client = None
        db = None

# --- History Logging Function (MongoDB) ---
def log_prediction_history(kuih_name, calories):
    """Logs a successful prediction to the history collection."""
    if not db_connection_ok: return
    try:
        history_collection = db.prediction_history
        cal_val = str(calories) if calories is not None else 'N/A'
        history_collection.insert_one({
            "kuih_name": kuih_name,
            "calories": cal_val,
            "timestamp": datetime.now()
        })
    except Exception as e:
        logger.error(f"Failed to log history to MongoDB: {e}")

# --- Get Kuih Details (MongoDB) ---
def get_kuih_details_from_db(kuih_name):
    if not db_connection_ok: return None
    details = None
    try:
        calories_collection = db.calories
        result = calories_collection.find_one({"kuih_name": kuih_name})
        
        if result:
            details = {
                'kuih_name': result.get('kuih_name'),
                'calories': int(result.get('calories')) if result.get('calories') is not None else 'N/A',
                'size_servings': result.get('size_servings', 'N/A'),
                'weight': str(result.get('weight', 'N/A')),
                'references': result.get('references', 'N/A'),
                'links': result.get('links', 'N/A')
            }
    except Exception as e:
        logger.error(f"DB error fetching details from MongoDB: {e}")
    return details

# --- Save Feedback (MongoDB) ---
def save_feedback_to_db(predicted_label, is_correct, actual_label=None, image_filename=None):
    if not db_connection_ok: return False
    
    image_db_path = None
    if image_filename:
        correct_label_for_path = actual_label if not is_correct else predicted_label
        if correct_label_for_path:
            image_db_path = os.path.join(Config.FEEDBACK_FOLDER, correct_label_for_path, image_filename)

    try:
        feedback_collection = db.feedback_log
        feedback_collection.insert_one({
            "predicted_label": predicted_label,
            "is_correct": 1 if is_correct else 0,
            "actual_label": actual_label,
            "timestamp": datetime.now(),
            "image_path": image_db_path
        })
        logger.info(f"Feedback saved to DB: Image='{image_filename}', Predicted='{predicted_label}', Correct={is_correct}, Actual='{actual_label}'")
        return True
    except Exception as err:
        logger.error(f"DB error saving feedback to MongoDB: {err}")
        return False

# --- Get Feedback Stats (MongoDB) ---
def get_feedback_stats():
    stats = {'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0}
    if not db_connection_ok: return stats
    
    try:
        feedback_collection = db.feedback_log
        stats['total'] = feedback_collection.count_documents({})
        if stats['total'] > 0:
            stats['correct'] = feedback_collection.count_documents({"is_correct": 1})
            stats['incorrect'] = stats['total'] - stats['correct']
            stats['accuracy'] = (stats['correct'] / stats['total'] * 100)
    except Exception as err:
        logger.error(f"DB error getting feedback stats from MongoDB: {err}")
    return stats

# --- Get Available Classes (MongoDB) ---
def get_available_classes_from_db():
    if not db_connection_ok: return class_labels
    try:
        calories_collection = db.calories
        classes = calories_collection.distinct("kuih_name")
        classes.sort()
        return classes if classes else class_labels
    except Exception as e:
        logger.error(f"DB error getting classes from MongoDB: {e}")
        return class_labels

# --- Model & Utils (Unchanged) ---
def load_trained_model():
    global model, model_loaded
    try:
        if os.path.exists(Config.MODEL_PATH):
            model = load_model(Config.MODEL_PATH)
            model_loaded = True
            logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Model load failed: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def predict_kuih(image_path):
    if not model_loaded: return "Model Error", 0.0
    try:
        img = load_img(image_path, target_size=Config.TARGET_SIZE)
        img_arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
        preds = model.predict(img_arr, verbose=0)
        idx = np.argmax(preds[0])
        return class_labels[idx], float(preds[0][idx])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "Prediction Error", 0.0

# --- Initialization ---
init_db() # <-- Initialize DB connection on startup
load_trained_model()

# --- Routes (Unchanged) ---
@app.route('/')
def home():
    return render_template('index.html',
                         model_loaded=model_loaded,
                         db_connection_ok=db_connection_ok,
                         feedback_stats=get_feedback_stats(),
                         available_classes=get_available_classes_from_db())

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename(filename))

@app.route('/predict', methods=['POST'])
def handle_predict():
    render_args = {
        'model_loaded': model_loaded,
        'db_connection_ok': db_connection_ok,
        'feedback_stats': get_feedback_stats(),
        'available_classes': get_available_classes_from_db()
    }

    if not model_loaded:
        return render_template('index.html', error="Model not loaded.", **render_args), 503

    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('index.html', error="No file selected.", **render_args), 400
    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type.", **render_args), 400

    try:
        fname = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)

        kuih_name, conf = predict_kuih(fpath)
        if "Error" in kuih_name:
             if os.path.exists(fpath): os.remove(fpath)
             return render_template('index.html', error=kuih_name, **render_args), 500

        details = get_kuih_details_from_db(kuih_name)
        
        calories_to_log = details['calories'] if details else 'N/A'
        log_prediction_history(kuih_name, calories_to_log)
        
        render_args.update({
            'success': True,
            'kuih_name': kuih_name,
            'confidence': f"{conf*100:.2f}%",
            'confidence_value': conf,
            'image_path': fname,
            'request_feedback': conf < Config.MIN_CONFIDENCE_THRESHOLD
        })

        if details:
            render_args.update(details)
        else:
             render_args.update({'calories': 'N/A', 'error_message': f"No details for {kuih_name}"})

        return render_template('index.html', **render_args)

    except Exception as e:
        logger.error(f"Predict route error: {e}")
        return render_template('index.html', error="Server error during prediction.", **render_args), 500

@app.route('/submit_feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    if not data: return jsonify({'success': False, 'message': 'No data'}), 400
    
    saved = save_feedback_to_db(
        data.get('predicted_label'),
        data.get('is_correct'),
        data.get('actual_label'),
        data.get('image_path')
    )
    return jsonify({'success': saved, 'message': 'Feedback saved!' if saved else 'Database error.'})

# --- NEW: Gemini AI Route ---
@app.route('/gemini-info', methods=['POST'])
def get_gemini_info():
    """Get description and fun fact from Gemini."""
    if not GEMINI_AVAILABLE:
        logger.error("Gemini route called, but Gemini is not available (check API key).")
        return jsonify({"error": "AI service is not configured."}), 503

    try:
        data = request.get_json()
        kuih_name = data.get('kuih')
        if not kuih_name:
            return jsonify({"error": "No kuih name provided."}), 400
        
        logger.info(f"Requesting Gemini info for: {kuih_name}")

        # --- Set up the model with the JSON schema ---
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_JSON_SCHEMA
        )
        model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-09-2025",
            generation_config=generation_config
        )
        
        # --- Create the prompt ---
        prompt = f"""
        You are a Malaysian food expert. 
        Provide estimation calories, serving and the sources and others names and a 2-3 sentence description and one interesting fun fact for the following Malaysian kuih: {kuih_name}.
        Ensure the fun fact is different from the description.
        """
        
        # --- Call the API ---
        response = model.generate_content(prompt)
        
        # --- Parse the JSON response ---
        # The response.text will be a JSON string that matches the schema
        response_data = json.loads(response.text)
        
        logger.info(f"Successfully got Gemini info for: {kuih_name}")
        return jsonify(response_data)

    except Exception as e:
        logger.exception(f"Error calling Gemini API: {e}")
        return jsonify({"error": "Failed to get AI insights."}), 500
# --- END NEW ---

@app.route('/api/history')
def get_history():
    history_data = []
    if db_connection_ok:
        try:
            history_collection = db.prediction_history
            history_data = list(history_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(50))
        except Exception as e:
             logger.error(f"History fetch error from MongoDB: {e}")
    
    return jsonify(history_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)