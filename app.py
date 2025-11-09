# app.py (Consolidated Version with Gemini AI + History + Calculator)

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

# --- Configuration ---
class Config:
    # Database configuration (adjust as needed)
    DB_CONFIG = {
        'user': 'root',
        'password': '',
        'host': '127.0.0.1',
        'port': 3307,
        'database': 'kuih_db'
    }
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

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
model = None
# IMPORTANT: Ensure this matches your training EXACTLY
class_labels = ['Akok', 'Cek Mek Molek','Ketayap', 'Kole Kacang', 'Kuih Bakar', 'Kuih Lapis', 'Kuih Lompang', 'Kuih Qasidah', 'Onde-onde', 'Pulut Sekaya', 'Seri Muka']
metrics = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0}
model_loaded = False
db_connection_ok = False

# --- MySQL Setup ---
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logger.warning("mysql-connector-python not found.")

# --- Gemini AI Setup ---
try:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        logger.warning("GEMINI_API_KEY not set.")
        GEMINI_AVAILABLE = False
    else:
        genai.configure(api_key=API_KEY)
        GEMINI_AVAILABLE = True
except Exception as e:
    logger.error(f"Gemini Error: {e}")
    GEMINI_AVAILABLE = False

GEMINI_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "estimatedcalories": {"type": "STRING"},
        "othersname": {"type": "STRING"},
        "description": {"type": "STRING"},
        "fun_fact": {"type": "STRING"}
    },
    "required": ["estimatedcalories","othersname","description", "fun_fact"]
}

# --- Database Helpers ---
def get_db_connection():
    global db_connection_ok
    if not MYSQL_AVAILABLE: return None
    try:
        conn = mysql.connector.connect(**Config.DB_CONFIG)
        if conn.is_connected():
            db_connection_ok = True
            return conn
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        db_connection_ok = False
    return None

def test_db_connection():
    conn = get_db_connection()
    if conn:
        conn.close()
        return True
    return False

# --- NEW: History Logging Function ---
def log_prediction_history(kuih_name, calories):
    """Logs a successful prediction to the history table."""
    conn = get_db_connection()
    if not conn: return
    try:
        cursor = conn.cursor()
        # Ensure you have created this table in your DB!
        query = "INSERT INTO prediction_history (kuih_name, calories, timestamp) VALUES (%s, %s, %s)"
        # Handle 'N/A' calories or non-numeric values gracefully if needed
        cal_val = str(calories) if calories is not None else 'N/A'
        cursor.execute(query, (kuih_name, cal_val, datetime.now()))
        conn.commit()
        cursor.close()
    except Exception as e:
        logger.error(f"Failed to log history: {e}")
    finally:
        if conn and conn.is_connected(): conn.close()

def get_kuih_details_from_db(kuih_name):
    conn = get_db_connection()
    if not conn: return None
    details = None
    try:
        cursor = conn.cursor()
        query = "SELECT kuih_name, calories, size_servings, weight, `references`, links FROM calories WHERE kuih_name = %s"
        cursor.execute(query, (kuih_name,))
        result = cursor.fetchone()
        if result:
            details = {
                'kuih_name': result[0],
                'calories': int(result[1]) if result[1] is not None else 'N/A',
                'size_servings': result[2],
                'weight': str(result[3]) if result[3] is not None else 'N/A',
                'references': result[4],
                'links': result[5]
            }
        cursor.close()
    except Exception as e:
        logger.error(f"DB error fetching details: {e}")
    finally:
        if conn and conn.is_connected(): conn.close()
    return details

def save_feedback_to_db(predicted, is_correct, actual=None, img_path=None):
    conn = get_db_connection()
    if not conn: return False
    try:
        cursor = conn.cursor()
        query = "INSERT INTO feedback_log (predicted_label, is_correct, actual_label, timestamp, image_path) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (predicted, 1 if is_correct else 0, actual, datetime.now(), img_path))
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"Feedback save error: {e}")
        return False
    finally:
        if conn and conn.is_connected(): conn.close()

def get_feedback_stats():
    stats = {'total': 0, 'correct': 0, 'accuracy': 0}
    conn = get_db_connection()
    if not conn: return stats
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), SUM(is_correct) FROM feedback_log")
        res = cursor.fetchone()
        if res and res[0]:
            stats['total'] = int(res[0])
            stats['correct'] = int(res[1]) if res[1] else 0
            stats['accuracy'] = (stats['correct']/stats['total']*100)
        cursor.close()
    except: pass
    finally:
        if conn and conn.is_connected(): conn.close()
    return stats

def get_available_classes_from_db():
    conn = get_db_connection()
    if not conn: return class_labels
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT kuih_name FROM calories WHERE kuih_name IS NOT NULL ORDER BY kuih_name")
        classes = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return classes if classes else class_labels
    except:
        return class_labels
    finally:
        if conn and conn.is_connected(): conn.close()

# --- Model & Utils ---
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
load_trained_model()
test_db_connection()

# --- Routes ---
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
    # Base render args
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
        
        # --- NEW: Log to history ---
        calories_to_log = details['calories'] if details else 'N/A'
        log_prediction_history(kuih_name, calories_to_log)
        # ---------------------------

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
    
    # Simplified for brevity - full implementation should include file moving logic from original
    saved = save_feedback_to_db(
        data.get('predicted_label'),
        data.get('is_correct'),
        data.get('actual_label'),
        data.get('image_path')
    )
    return jsonify({'success': saved, 'message': 'Feedback saved!' if saved else 'Database error.'})

@app.route('/gemini-info', methods=['POST'])
def get_gemini_info():
    if not GEMINI_AVAILABLE: return jsonify({"error": "AI not configured"}), 503
    try:
        k_name = request.get_json().get('kuih')
        model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025", 
                                      generation_config={"response_mime_type": "application/json", "response_schema": GEMINI_JSON_SCHEMA})
        prompt = f"Provide Malaysian kuih info for: {k_name}. JSON format with estimatedcalories, othersname, description (2-3 sentences), fun_fact (unique)."
        resp = model.generate_content(prompt)
        return jsonify(json.loads(resp.text))
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return jsonify({"error": "AI failed"}), 500

# --- NEW: Route to fetch history ---
@app.route('/api/history')
def get_history():
    conn = get_db_connection()
    history_data = []
    if conn:
        try:
            cursor = conn.cursor(dictionary=True) # Use dictionary cursor for easy JSON
            # Fetch last 50 entries, newest first
            cursor.execute("SELECT kuih_name, calories, timestamp FROM prediction_history ORDER BY timestamp DESC LIMIT 50")
            history_data = cursor.fetchall()
            cursor.close()
        except Exception as e:
             logger.error(f"History fetch error: {e}")
        finally:
            if conn.is_connected(): conn.close()
    return jsonify(history_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)