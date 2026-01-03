import os
import string
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from threading import Lock
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__)
app.secret_key = 'super_secret_key_change_this'

# --- 1. CONFIGURATION & ENCODER ---
ALL_CHARS = string.ascii_uppercase + string.ascii_lowercase + string.digits
CHAR_TO_INDEX = {char: i for i, char in enumerate(ALL_CHARS)}
INDEX_TO_CHAR = {i: char for i, char in enumerate(ALL_CHARS)}
NUM_CLASSES = 62

DATA_DIR = 'data'
LABELS_FILE = os.path.join(DATA_DIR, 'labels.npy')
IMAGES_FILE = os.path.join(DATA_DIR, 'images.npy')
MODEL_FILE = 'letter.keras'

data_lock = Lock()

# --- 2. GLOBAL MODEL LOADING ---
MODEL = None

def load_global_model():
    global MODEL
    if os.path.exists(MODEL_FILE):
        try:
            MODEL = keras.models.load_model(MODEL_FILE)
            print(">> Model loaded successfully.")
        except Exception as e:
            print(f">> Error loading model: {e}")
    else:
        print(">> No model found.")

# Initial load
load_global_model()

# --- 3. INITIALIZATION HELPER ---
def init_data_files():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(LABELS_FILE):
        np.save(LABELS_FILE, np.array([]))
    if not os.path.exists(IMAGES_FILE):
        np.save(IMAGES_FILE, np.empty((0, 50, 50)))

init_data_files()

# --- 4. INTERNAL TRAINING FUNCTION ---
def train_internal():
    # 1. Load Data
    if not os.path.exists(LABELS_FILE): return False, "No data files found."
    
    labels_raw = np.load(LABELS_FILE)
    images = np.load(IMAGES_FILE)

    if len(labels_raw) < 5: # Require at least 5 examples
        return False, "Not enough data. Please draw at least 5 characters first."

    # 2. Preprocess
    # Reshape to (N, 50, 50, 1) and normalize
    X_train = images.reshape(-1, 50, 50, 1).astype('float32') / 255.0
    y_train = np.array([CHAR_TO_INDEX[c] for c in labels_raw])

    # 3. Build Model
    model = keras.Sequential([
        layers.Input(shape=(50, 50, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 4. Train
    # Using fewer epochs (5) so the user doesn't wait too long
    model.fit(X_train, y_train, epochs=5, batch_size=4, verbose=1)

    # 5. Save & Reload
    model.save(MODEL_FILE)
    load_global_model() # Reload the global variable
    
    return True, f"Training complete on {len(labels_raw)} examples!"


# --- 5. ROUTES ---

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def train_route():
    # Trigger the training logic
    success, message = train_internal()
    if success:
        session['message'] = message
        # Go directly to practice page if successful
        return redirect(url_for('practice_get'))
    else:
        session['error'] = message
        return redirect(url_for('add_data_get'))

@app.route('/add-data', methods=['GET'])
def add_data_get():
    message = session.pop('message', '')
    error = session.pop('error', '')
    target_char = np.random.choice(list(ALL_CHARS))
    return render_template("addData.html", letter=target_char, message=message, error=error)

@app.route('/add-data', methods=['POST'])
def add_data_post():
    label_char = request.form['letter']
    pixels = request.form['pixels']
    
    if pixels:
        pixels_list = pixels.split(',')
        img_array = np.array(pixels_list).astype(float).reshape(1, 50, 50)

        with data_lock:
            current_labels = np.load(LABELS_FILE)
            current_images = np.load(IMAGES_FILE)
            new_labels = np.append(current_labels, label_char)
            new_images = np.vstack([current_images, img_array])
            np.save(LABELS_FILE, new_labels)
            np.save(IMAGES_FILE, new_images)
        
        session['message'] = f'Saved "{label_char}". Keep going!'
    
    return redirect(url_for('add_data_get'))

@app.route('/practice', methods=['GET'])
def practice_get():
    message = session.pop('message', '')
    target_char = np.random.choice(list(ALL_CHARS))
    return render_template("practice.html", letter=target_char, message=message)

@app.route('/practice', methods=['POST'])
def practice_post():
    target_char = request.form['letter']
    pixels = request.form['pixels']
    
    if not MODEL:
        return render_template("practice.html", letter=target_char, 
                               error="Model not trained yet! Click 'Train Model' in the menu.")

    img = np.array(pixels.split(',')).astype(float).reshape(1, 50, 50, 1)
    # img = img / 255.0 # Add this if you normalized in training

    preds = MODEL.predict(img)
    pred_index = np.argmax(preds, axis=-1)[0]
    pred_char = INDEX_TO_CHAR[pred_index]
    confidence = np.max(preds) * 100

    result = {
        'correct': (pred_char == target_char),
        'pred_char': pred_char,
        'confidence': round(confidence, 1),
        'target': target_char
    }
    
    return render_template("practice.html", letter=np.random.choice(list(ALL_CHARS)), result=result)

if __name__=='__main__':
    app.run(debug=True, port=5000)