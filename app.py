from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import os
import logging
import atexit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32).hex())
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])
except ImportError:
    limiter = None
    logger.warning("flask-limiter not installed, rate limiting disabled")


def _limit(limit_string):
    if limiter:
        return limiter.limit(limit_string)
    return lambda f: f


@app.after_request
def set_security_headers(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "img-src 'self' data:; "
    )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'model.h5')

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model from %s: %s", MODEL_PATH, e)
    model = None

class_labels = ['Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    logger.error("Failed to load Haar cascade for face detection")

camera = None


def cleanup_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        logger.info("Camera released on shutdown")


atexit.register(cleanup_camera)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def detect_faces_and_emotions(image):
    if model is None:
        return image, "Model not loaded"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image, "No face detected"

    detected_emotion = None
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face = image[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (96, 96))
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_resized = img_to_array(face_resized)
        face_resized = np.expand_dims(face_resized, axis=0) / 255.0

        prediction = model.predict(face_resized, verbose=0)
        max_index = np.argmax(prediction[0])
        detected_emotion = class_labels[max_index]

        label_position = (x, y - 10)
        cv2.putText(image, detected_emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image, detected_emotion


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
@_limit("10 per minute")
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('upload.html', emotion=None, error='No file selected')

        file = request.files['image']
        if not file or not file.filename:
            return render_template('upload.html', emotion=None, error='No file selected')

        if not allowed_file(file.filename):
            return render_template('upload.html', emotion=None,
                                   error='Unsupported file format. Please upload JPG, PNG, BMP, or WebP.')

        try:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                return render_template('upload.html', emotion=None, error='Could not read image. Please try another file.')

            processed_image, emotion = detect_faces_and_emotions(image)

            _, buffer = cv2.imencode('.jpg', processed_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template('upload.html', emotion=emotion, image=image_base64)

        except Exception as e:
            logger.exception("Error processing uploaded image")
            return render_template('upload.html', emotion=None, error='Error processing image. Please try a different file.')

    return render_template('upload.html', emotion=None)


@app.route('/real_time')
def real_time():
    return render_template('real_time.html', stream=False)


@app.route('/start', methods=['POST'])
def start_detection():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

    return render_template('real_time.html', stream=True)


@app.route('/video_feed')
def video_feed():
    global camera

    if camera is None:
        camera = cv2.VideoCapture(0)

    def gen_frames():
        while camera is not None:
            success, frame = camera.read()
            if not success:
                break

            frame, _ = detect_faces_and_emotions(frame)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop', methods=['POST'])
def stop_detection():
    global camera
    if camera:
        camera.release()
        camera = None

    return redirect(url_for('real_time'))


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'cascade_loaded': not face_cascade.empty()
    })


@app.errorhandler(413)
def file_too_large(e):
    return render_template('upload.html', emotion=None, error='File too large. Maximum size is 16MB.'), 413


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    debug = os.environ.get('FLASK_ENV') != 'production'
    host = '127.0.0.1'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, host=host, port=port)
