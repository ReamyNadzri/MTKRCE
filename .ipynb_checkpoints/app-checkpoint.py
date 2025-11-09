# app.py (Consolidated Version with Gemini AI)

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
import shutil # For moving feedback images
import json # --- NEW: Added for Gemini JSON parsing ---

# --- NEW: Added for Gemini AI ---
import google.generativeai as genai
# --- END NEW ---


# Optional MySQL import
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    print("WARNING: mysql-connector-python not found. Database features will be disabled.")

# Optional sklearn import (only needed if calculating metrics on load)
# try:
#     from sklearn.metrics import classification_report
#     SKLEARN_AVAILABLE = True
# except ImportError:
#     SKLEARN_AVAILABLE = False
#     print("WARNING: scikit-learn not found. Model evaluation metrics display might be limited.")


# --- Setup Logging ---
# Use a format that includes timestamps for better debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    # Database configuration (adjust as needed)
    DB_CONFIG = {
        'user': 'root',
        'password': '',
        'host': '127.0.0.1', # Or your DB host
        'port': 3307,        # Default MySQL port is 3306, adjust if needed
        'database': 'kuih_db' # Your database name
    }

    # File upload configuration
    UPLOAD_FOLDER = 'uploads'
    FEEDBACK_FOLDER = 'feedback_images'  # Store images with feedback
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

    # Model configuration
    MODEL_PATH = 'kuih_recognition_model.keras'
    TARGET_SIZE = (224, 224)
    # IMAGES_PATH = 'data_split/test' # Needed only if calculating metrics on load

    # Feedback configuration
    MIN_CONFIDENCE_THRESHOLD = 0.7

# --- Flask App Setup ---
app = Flask(__name__)
app.config.from_object(Config)

# Ensure directories exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.FEEDBACK_FOLDER, exist_ok=True)

# --- Global Variables ---
model = None
# IMPORTANT: Update this list to EXACTLY match the class order from your training notebook output
class_labels = ['Akok', 'Cek Mek Molek', 'Kole Kacang', 'Kuih Bakar', 'Kuih Lompang', 'Kuih Qasidah', 'Pulut Sekaya', 'Seri Muka'] # Add the other 2 when model is retrained
metrics = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0} # Placeholder
model_loaded = False
db_connection_ok = False

# --- NEW: Gemini AI Configuration ---
# --- IMPORTANT: SET YOUR API KEY IN YOUR TERMINAL ---
# --- On Mac/Linux: export GEMINI_API_KEY="YOUR_API_KEY_HERE"
# --- On Windows: set GEMINI_API_KEY="YOUR_API_KEY_HERE"
# --- DO NOT paste your key directly into this file.
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

# --- NEW: Define the JSON structure we want Gemini to return ---
GEMINI_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "description": {
            "type": "STRING",
            "description": "A brief, engaging 2-3 sentence description of the kuih."
        },
        "fun_fact": {
            "type": "STRING",
            "description": "A single interesting fun fact about the kuih's history, ingredients, or cultural significance."
        }
    },
    "required": ["description", "fun_fact"]
}
# --- END NEW ---


# --- Database Connection Test ---
def test_db_connection():
    """Tests the database connection defined in Config."""
    global db_connection_ok
    if not MYSQL_AVAILABLE:
        logger.warning("MySQL connector not available. Skipping DB connection test.")
        db_connection_ok = False
        return False
    conn = None
    try:
        conn = mysql.connector.connect(**Config.DB_CONFIG, connection_timeout=5)
        if conn.is_connected():
            logger.info(f"Successfully connected to database '{Config.DB_CONFIG['database']}'.")
            db_connection_ok = True
            return True
        else:
            logger.error(f"Failed to connect to database '{Config.DB_CONFIG['database']}'.")
            db_connection_ok = False
            return False
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        db_connection_ok = False
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()

# --- Utility Functions ---
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def create_unique_filename(filename):
    """Create a unique filename to prevent conflicts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(secure_filename(filename))
    return f"{name}_{timestamp}_{unique_id}{ext}"

def get_db_connection():
    """Establishes and returns a database connection if available and tested OK."""
    global db_connection_ok # <-- MOVED TO THE TOP of the function

    if not MYSQL_AVAILABLE or not db_connection_ok:
        logger.warning("Attempted DB operation but connection unavailable or failed.")
        return None
    try:
        conn = mysql.connector.connect(**Config.DB_CONFIG)
        # Check connection again immediately after connecting (optional but good)
        if not conn.is_connected():
            logger.error("DB connection established but immediately disconnected.")
            db_connection_ok = False # Update status
            return None
        return conn # Return the active connection
    except mysql.connector.Error as err:
        logger.error(f"Database connection failed during operation: {err}")
        # Mark connection as not OK if it fails during operation
        db_connection_ok = False
        return None

def get_calories_from_db(kuih_name):
    """Get calorie information from the database."""
    calories = "N/A"
    conn = get_db_connection()
    if conn is None:
        return "DB Error" # Indicate DB problem
    try:
        cursor = conn.cursor()
        query = "SELECT calories FROM calories WHERE kuih_name = %s"
        cursor.execute(query, (kuih_name,))
        result = cursor.fetchone()
        if result and result[0] is not None: # Check if result exists and calorie is not NULL
             # Format as integer if possible, otherwise string
            try:
                calories = str(int(result[0]))
            except (ValueError, TypeError):
                calories = str(result[0])
        else:
            logger.warning(f"No calorie data found for '{kuih_name}' in DB.")
        cursor.close()
    except mysql.connector.Error as err:
        logger.error(f"DB error fetching calories for '{kuih_name}': {err}")
        calories = "DB Error"
    finally:
        if conn and conn.is_connected():
            conn.close()
    return calories

def save_feedback_to_db(predicted_label, is_correct, actual_label=None, image_filename=None):
    """Save user feedback to the database."""
    conn = get_db_connection()
    if conn is None:
        return False # DB connection failed
    
    # Determine the final path where the image *should* be after moving
    image_db_path = None
    if image_filename:
        correct_label_for_path = actual_label if not is_correct else predicted_label
        if correct_label_for_path: # Ensure we have a label to form the path
            image_db_path = os.path.join(Config.FEEDBACK_FOLDER, correct_label_for_path, image_filename)

    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO feedback_log (predicted_label, is_correct, actual_label, timestamp, image_path)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            predicted_label,
            1 if is_correct else 0,
            actual_label,
            datetime.now(),
            image_db_path # Store the potential final path in feedback folder
        ))
        conn.commit()
        cursor.close()
        logger.info(f"Feedback saved to DB: Image='{image_filename}', Predicted='{predicted_label}', Correct={is_correct}, Actual='{actual_label}'")
        return True
    except mysql.connector.Error as err:
        logger.error(f"DB error saving feedback: {err}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()

def get_feedback_stats():
    """Get statistics about collected feedback from DB."""
    stats = {'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0}
    conn = get_db_connection()
    if conn is None:
        return stats # Return default stats if DB unavailable
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), SUM(is_correct) FROM feedback_log")
        result = cursor.fetchone()
        if result and result[0] is not None and result[0] > 0:
            stats['total'] = int(result[0])
            stats['correct'] = int(result[1]) if result[1] is not None else 0
            stats['incorrect'] = stats['total'] - stats['correct']
            stats['accuracy'] = (stats['correct'] / stats['total'] * 100)
        cursor.close()
    except mysql.connector.Error as err:
        logger.error(f"DB error getting feedback stats: {err}")
    finally:
        if conn and conn.is_connected():
            conn.close()
    return stats

def get_available_classes_from_db():
    """Get list of distinct kuih names from the calories table for the dropdown."""
    db_classes = []
    conn = get_db_connection()
    # Fallback to model's labels if DB unavailable or empty
    fallback_classes = class_labels if class_labels else []
    if conn is None:
        logger.warning("DB unavailable for getting classes, using model labels.")
        return fallback_classes

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT kuih_name FROM calories WHERE kuih_name IS NOT NULL ORDER BY kuih_name")
        db_classes = [row[0] for row in cursor.fetchall()]
        cursor.close()
        if not db_classes:
             logger.warning("No classes found in DB 'calories' table, using model labels.")
             return fallback_classes
        else:
             # logger.info(f"Fetched {len(db_classes)} classes from DB for feedback dropdown.")
             return db_classes

    except mysql.connector.Error as err:
        logger.error(f"DB error getting available classes: {err}")
        return fallback_classes # Fallback
    finally:
        if conn and conn.is_connected():
            conn.close()

def save_feedback_image_file(original_filename, predicted_label, actual_label):
    """Moves the uploaded image to the feedback folder structure based on ACTUAL label."""
    source_path = os.path.join(Config.UPLOAD_FOLDER, original_filename)
    if not os.path.exists(source_path):
        logger.warning(f"Source image '{original_filename}' not found in uploads for feedback.")
        return None # Indicate file wasn't found/moved

    try:
        # Target directory is based on the *actual* (or confirmed correct) label
        target_class_dir = os.path.join(Config.FEEDBACK_FOLDER, actual_label)
        os.makedirs(target_class_dir, exist_ok=True)

        # Keep the unique filename generated during upload
        feedback_filename = original_filename
        target_path = os.path.join(target_class_dir, feedback_filename)

        # Handle potential file collision (though unlikely with unique names)
        if os.path.exists(target_path):
            logger.warning(f"Feedback file already exists: {target_path}. Overwriting.")
            # Or generate a new name variant if overwrite is not desired

        # Move the file
        shutil.move(source_path, target_path)
        logger.info(f"Moved feedback image from '{source_path}' to '{target_path}'")
        return feedback_filename # Return the filename that was moved for DB logging

    except Exception as e:
        logger.error(f"Error moving feedback image '{original_filename}' to class '{actual_label}': {e}")
        # If move fails, should we delete the source? Let's leave it for manual check.
        # Consider adding retry logic or specific error handling (permissions etc.)
        return None # Indicate move failed


def load_trained_model():
    """Loads the trained Keras model and sets class labels."""
    global model, class_labels, model_loaded, metrics
    try:
        if not os.path.exists(Config.MODEL_PATH):
            logger.error(f"Model file not found: {Config.MODEL_PATH}.")
            return False

        model = load_model(Config.MODEL_PATH)
        logger.info(f"Keras model loaded successfully from '{Config.MODEL_PATH}'")

        # Class labels are critical and MUST match training order.
        # Hardcoding is okay if the training order is fixed and known.
        # Ensure this list is updated if the model is retrained with different classes/order.
        logger.info(f"Using defined class labels (ensure order matches training): {class_labels}")
        if not class_labels:
             logger.error("Class labels are empty! Model cannot function correctly.")
             return False


        # Optional: Calculate metrics on load (requires test set & sklearn)
        # Add this section back if needed and SKLEARN_AVAILABLE is True

        model_loaded = True
        return True
    except Exception as e:
        logger.error(f"Error loading Keras model: {e}")
        return False

def predict_kuih(image_path):
    """Makes prediction on an image using the loaded model."""
    if not model_loaded or model is None:
        logger.error("Model not loaded, cannot predict.")
        return "Model Error", 0.0

    if not class_labels:
        logger.error("Class labels not defined, cannot map prediction.")
        return "Label Error", 0.0

    try:
        # Load and preprocess image matching training (rescale 1./255)
        img = load_img(image_path, target_size=Config.TARGET_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Create batch
        img_preprocessed = img_array / 255.0

        # Make prediction
        predictions = model.predict(img_preprocessed, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])

        # Map index to label
        if 0 <= predicted_class_index < len(class_labels):
            predicted_kuih = class_labels[predicted_class_index]
            logger.info(f"Prediction successful: {predicted_kuih} (Confidence: {confidence:.4f})")
            return predicted_kuih, confidence
        else:
            logger.error(f"Predicted class index {predicted_class_index} is out of range for {len(class_labels)} labels.")
            return "Index Error", 0.0 # Specific error for index issue

    except Exception as e:
        logger.exception(f"Exception during prediction for {image_path}: {e}") # Log full traceback
        return "Prediction Error", 0.0 # General prediction error

# --- Load Model and Test DB on Startup ---
logger.info("Initializing application...")
load_trained_model()
test_db_connection() # Sets db_connection_ok flag

# --- Flask Routes ---
@app.route('/')
def home():
    """Home page route."""
    # Fetch fresh data for each page load
    feedback_stats = get_feedback_stats()
    available_classes_db = get_available_classes_from_db()
    return render_template('index.html',
                         metrics=metrics,
                         model_loaded=model_loaded,
                         db_connection_ok=db_connection_ok,
                         class_labels=class_labels, # Model's trained classes
                         feedback_stats=feedback_stats,
                         available_classes=available_classes_db # For feedback dropdown
                         )

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    """Serves uploaded images."""
    # Basic security check (optional): ensure filename is safe
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        logger.warning(f"Attempt to access potentially unsafe path: {filename}")
        return "Not Found", 404 # Or redirect, or stricter check
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename)

@app.route('/predict', methods=['POST'])
def handle_predict():
    """Handle image prediction requests and renders HTML."""
    # Fetch fresh stats and classes for rendering, even in error cases
    feedback_stats = get_feedback_stats()
    available_classes_db = get_available_classes_from_db()
    render_args = {
        'metrics': metrics,
        'model_loaded': model_loaded,
        'db_connection_ok': db_connection_ok,
        'class_labels': class_labels,
        'feedback_stats': feedback_stats,
        'available_classes': available_classes_db
    }

    if not model_loaded:
        render_args['error'] = "Model is not loaded. Cannot predict."
        return render_template('index.html', **render_args), 503 # Service Unavailable

    file = request.files.get('file')

    if not file or file.filename == '':
        render_args['error'] = "No file selected."
        return render_template('index.html', **render_args), 400

    if not allowed_file(file.filename):
        render_args['error'] = f"Invalid file type. Allowed: {Config.ALLOWED_EXTENSIONS}"
        return render_template('index.html', **render_args), 400

    file_path = None # Define outside try for cleanup
    try:
        unique_filename = create_unique_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        logger.info(f"File saved temporarily to {file_path}")

        kuih_name, confidence = predict_kuih(file_path)

        # Check for prediction specific errors
        if kuih_name in ["Model Error", "Prediction Error", "Index Error", "Label Error"]:
             render_args['error'] = f"Prediction failed: {kuih_name}"
             # Clean up failed prediction upload
             if os.path.exists(file_path): os.remove(file_path)
             return render_template('index.html', **render_args), 500

        calories = get_calories_from_db(kuih_name)
        request_feedback = confidence < Config.MIN_CONFIDENCE_THRESHOLD

        # Add successful prediction results to render_args
        render_args.update({
            'kuih_name': kuih_name,
            'confidence': f"{confidence*100:.2f}%",
            'confidence_value': confidence,
            'calories': calories,
            'image_path': unique_filename, # Pass filename for display URL
            'request_feedback': request_feedback,
            'success': True # Flag for template to show results section
        })

        # Don't delete file yet, wait for potential feedback
        logger.info(f"Prediction successful for {unique_filename}. Results ready.")
        return render_template('index.html', **render_args)

    except Exception as e:
        logger.exception(f"Unexpected error in /predict route: {e}") # Log full traceback
        render_args['error'] = "An unexpected server error occurred during prediction."
        # Clean up saved file if an unexpected error occurred
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up {file_path} after unexpected error.")
            except OSError:
                logger.warning(f"Could not remove file after unexpected error: {file_path}")
        return render_template('index.html', **render_args), 500


@app.route('/submit_feedback', methods=['POST'])
def handle_feedback():
    """Handle user feedback submission via JSON."""
    if not model_loaded or not db_connection_ok:
        logger.warning("Feedback received but system not ready (Model/DB).")
        return jsonify({'success': False, 'message': 'System not ready for feedback.'}), 503

    moved_filename = None # Track if file move was attempted/successful
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'Invalid request data.'}), 400

        predicted_label = data.get('predicted_label')
        is_correct = data.get('is_correct')
        actual_label = data.get('actual_label') # Only present if is_correct is false
        image_filename = data.get('image_path') # The unique filename from uploads

        # Validate input
        if not all([predicted_label is not None, is_correct is not None, image_filename]):
            return jsonify({'success': False, 'message': 'Missing required feedback data.'}), 400
        if not isinstance(is_correct, bool):
             return jsonify({'success': False, 'message': 'Invalid value for is_correct.'}), 400
        if not is_correct and not actual_label:
            return jsonify({'success': False, 'message': 'Actual label required for incorrect predictions.'}), 400

        # Determine the correct label used for folder path and DB
        final_actual_label = actual_label if not is_correct else predicted_label

        # 1. Move image file to feedback folder structure
        moved_filename = save_feedback_image_file(image_filename, predicted_label, final_actual_label)

        # 2. Save feedback details to Database (only if file move was relevant/attempted/successful?)
        # Let's save to DB even if move failed, but maybe log path as NULL or original path
        feedback_saved = save_feedback_to_db(
            predicted_label,
            is_correct,
            actual_label if not is_correct else None,
            moved_filename # Pass the filename (or None if move failed/not applicable)
        )

        if feedback_saved:
            # Optionally trigger retraining check here later
            return jsonify({
                'success': True,
                'message': 'Thank you for your feedback!',
                'feedback_stats': get_feedback_stats() # Send updated stats
            })
        else:
            # DB save failed. What to do with the moved file? Log it.
            if moved_filename:
                moved_path = os.path.join(Config.FEEDBACK_FOLDER, final_actual_label, moved_filename)
                logger.error(f"DB save failed for feedback on {moved_filename}. File IS ALREADY MOVED to {moved_path}. Manual DB entry might be needed.")
            else:
                 logger.error(f"DB save failed for feedback on {image_filename}. Image was NOT moved successfully.")

            return jsonify({
                'success': False,
                'message': 'Could not save feedback to database.'
            }), 500

    except Exception as e:
        logger.exception(f"Unexpected error processing feedback: {e}") # Log full traceback
         # Attempt cleanup of original upload if file still exists and wasn't moved
        image_filename_from_data = data.get('image_path') if 'data' in locals() else None
        source_path = os.path.join(Config.UPLOAD_FOLDER, image_filename_from_data) if image_filename_from_data else None
        if source_path and os.path.exists(source_path) and not moved_filename:
            try:
                os.remove(source_path)
                logger.info(f"Cleaned up {source_path} after feedback processing error.")
            except OSError:
                 logger.warning(f"Could not remove {source_path} after feedback error.")

        return jsonify({'error': 'Server error processing feedback', 'success': False}), 500


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
        Provide a 2-3 sentence description and one interesting fun fact for the following Malaysian kuih: {kuih_name}.
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


# --- Health Check and Error Handlers ---
@app.route('/health')
def health_check():
    """Health check endpoint."""
    # Re-test DB connection for current status
    current_db_ok = test_db_connection()
    return jsonify({
        'status': 'healthy' if model_loaded and current_db_ok else 'degraded',
        'model_loaded': model_loaded,
        'db_connection_ok': current_db_ok,
        'mysql_available': MYSQL_AVAILABLE,
        'num_classes': len(class_labels) if class_labels else 0,
        'feedback_collected': get_feedback_stats()['total'],
        'gemini_available': GEMINI_AVAILABLE # --- NEW: Added Gemini status
    })

@app.errorhandler(413)
def too_large(e):
    # Render error within the main template
    feedback_stats = get_feedback_stats()
    available_classes_db = get_available_classes_from_db()
    render_args = { 'metrics': metrics, 'model_loaded': model_loaded, 'db_connection_ok': db_connection_ok, 'class_labels': class_labels, 'feedback_stats': feedback_stats, 'available_classes': available_classes_db }
    render_args['error'] = 'File too large. Maximum size is 16MB.'
    return render_template('index.html', **render_args), 413

@app.errorhandler(500)
def internal_error(e):
    logger.exception(f"Caught Internal Server Error: {e}") # Log full traceback for 500 errors
    feedback_stats = get_feedback_stats()
    available_classes_db = get_available_classes_from_db()
    render_args = { 'metrics': metrics, 'model_loaded': model_loaded, 'db_connection_ok': db_connection_ok, 'class_labels': class_labels, 'feedback_stats': feedback_stats, 'available_classes': available_classes_db }
    render_args['error'] = 'An internal server error occurred. Please check server logs.'
    return render_template('index.html', **render_args), 500

@app.errorhandler(404)
def page_not_found(e):
     # Optional: redirect to home or show a simple 404 page
    # Check if it's a 404 for a specific API route the frontend might call
    if request.path == '/gemini-info':
        logger.error("404 Error: /gemini-info route was called but is not defined (this shouldn't happen with the new code).")
        return jsonify({"error": "Not Found"}), 404
        
    feedback_stats = get_feedback_stats()
    available_classes_db = get_available_classes_from_db()
    render_args = { 'metrics': metrics, 'model_loaded': model_loaded, 'db_connection_ok': db_connection_ok, 'class_labels': class_labels, 'feedback_stats': feedback_stats, 'available_classes': available_classes_db }
    render_args['error'] = 'Page not found.'
    return render_template('index.html', **render_args), 404


# --- Run the App ---
if __name__ == '__main__':
    logger.info("Starting Flask application...")
    if not model_loaded:
        logger.warning("MODEL NOT LOADED. Predictions will fail.")
    if not db_connection_ok:
        logger.warning("INITIAL DATABASE CONNECTION FAILED. Calorie/Feedback features might fail.")
    if not GEMINI_AVAILABLE:
        logger.warning("GEMINI AI NOT AVAILABLE. AI insights will fail.")
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)