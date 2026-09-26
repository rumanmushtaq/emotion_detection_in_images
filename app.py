import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import base64
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
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
        "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
        "img-src 'self' data:; "
    )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


EMOTION_COLORS = {
    'angry': (0, 0, 255),
    'disgust': (0, 128, 0),
    'fear': (128, 0, 128),
    'happy': (0, 255, 0),
    'sad': (255, 0, 0),
    'surprise': (0, 255, 255),
    'neutral': (200, 200, 200),
}

COMPOUND_EMOTIONS = [
    {'name': 'Happily Surprised', 'requires': ('happy', 'surprise'), 'emoji': '🤩'},
    {'name': 'Sadly Angry', 'requires': ('sad', 'angry'), 'emoji': '😠'},
    {'name': 'Fearfully Surprised', 'requires': ('fear', 'surprise'), 'emoji': '😱'},
    {'name': 'Sadly Surprised', 'requires': ('sad', 'surprise'), 'emoji': '😧'},
    {'name': 'Angrily Disgusted', 'requires': ('angry', 'disgust'), 'emoji': '🤬'},
    {'name': 'Fearfully Angry', 'requires': ('fear', 'angry'), 'emoji': '😤'},
    {'name': 'Sadly Fearful', 'requires': ('sad', 'fear'), 'emoji': '😰'},
    {'name': 'Happily Disgusted', 'requires': ('happy', 'disgust'), 'emoji': '😏'},
    {'name': 'Contempt', 'requires': ('neutral', 'disgust'), 'emoji': '😒'},
    {'name': 'Awe', 'requires': ('fear', 'happy'), 'emoji': '😲'},
    {'name': 'Anxious', 'requires': ('fear', 'sad'), 'emoji': '😟'},
    {'name': 'Outraged', 'requires': ('angry', 'surprise'), 'emoji': '🤯'},
    {'name': 'Bored', 'requires': ('neutral', 'sad'), 'emoji': '😐'},
    {'name': 'Pleased', 'requires': ('happy', 'neutral'), 'emoji': '😊'},
    {'name': 'Horrified', 'requires': ('fear', 'disgust'), 'emoji': '😨'},
]

INTENSITY_LABELS = {
    'angry': {(0, 20): 'Slightly Irritated', (20, 50): 'Annoyed', (50, 75): 'Angry', (75, 100): 'Furious'},
    'disgust': {(0, 20): 'Mildly Disgusted', (20, 50): 'Disgusted', (50, 75): 'Revolted', (75, 100): 'Appalled'},
    'fear': {(0, 20): 'Uneasy', (20, 50): 'Worried', (50, 75): 'Afraid', (75, 100): 'Terrified'},
    'happy': {(0, 20): 'Content', (20, 50): 'Pleased', (50, 75): 'Happy', (75, 100): 'Elated'},
    'sad': {(0, 20): 'Melancholic', (20, 50): 'Unhappy', (50, 75): 'Sad', (75, 100): 'Devastated'},
    'surprise': {(0, 20): 'Intrigued', (20, 50): 'Surprised', (50, 75): 'Amazed', (75, 100): 'Astonished'},
    'neutral': {(0, 20): 'Neutral', (20, 50): 'Calm', (50, 75): 'Composed', (75, 100): 'Stoic'},
}


def get_intensity_label(emotion, score):
    ranges = INTENSITY_LABELS.get(emotion, {})
    for (low, high), label in ranges.items():
        if low <= score < high:
            return label
    return emotion.capitalize()


def detect_compound_emotion(emotions):
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_score = sorted_emotions[0]
    top2_name, top2_score = sorted_emotions[1]

    if top2_score >= 15:
        for compound in COMPOUND_EMOTIONS:
            pair = compound['requires']
            if (top1_name, top2_name) == pair or (top2_name, top1_name) == pair:
                return compound['name'], compound['emoji']

    return get_intensity_label(top1_name, top1_score), ''


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
    try:
        results = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False, silent=True)
    except Exception as e:
        logger.error("DeepFace analysis failed: %s", e)
        return image, "Analysis failed", {}, '', ''

    if not results:
        return image, "No face detected", {}, '', ''

    all_emotions = {}
    dominant_emotion = None
    compound_label = ''
    compound_emoji = ''

    for face in results:
        region = face.get('region', {})
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)

        dominant_emotion = face.get('dominant_emotion', 'unknown')
        emotions = face.get('emotion', {})
        all_emotions = emotions

        compound_label, compound_emoji = detect_compound_emotion(emotions)

        color = EMOTION_COLORS.get(dominant_emotion, (0, 255, 0))

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        label = f"{compound_label} ({emotions.get(dominant_emotion, 0):.1f}%)"
        label_position = (x, y - 10)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image, dominant_emotion, all_emotions, compound_label, compound_emoji


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

            processed_image, emotion, emotions, compound, emoji = detect_faces_and_emotions(image)

            _, buffer = cv2.imencode('.jpg', processed_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template('upload.html', emotion=emotion, emotions=emotions,
                                   compound=compound, emoji=emoji, image=image_base64)

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

            frame, _, _, _, _ = detect_faces_and_emotions(frame)

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
        'emotions_supported': list(EMOTION_COLORS.keys())
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
