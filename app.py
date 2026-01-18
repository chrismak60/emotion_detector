"""
Emotion Detector Flask Application
Handles user authentication, image upload, emotion detection, and dashboard analytics
"""
from flask import Flask, session, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
import io
import base64
import cv2
import numpy as np
from helpers import recognize_emotion, generate_mood_content
from PIL import Image
import tempfile
from datetime import datetime
import uuid

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-emotion-detector'

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Redirect to login if not authenticated, otherwise to main app"""
    if 'username' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login with hardcoded credentials (user/54321)"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        # Validate credentials
        if username == 'user' and password == '54321':
            session['username'] = username
            session['emotion_history'] = []  # Initialize emotion history for this session
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Clear session and redirect to login"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/index')
def index():
    """Main app page - image upload and emotion detection"""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/dashboards')
def dashboards():
    """Analytics dashboard with charts and history"""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboards.html')

@app.route('/api/process-emotion', methods=['POST'])
def process_emotion():
    """
    Main API endpoint for emotion detection
    1. Receives uploaded image
    2. Detects face and emotion using DeepFace
    3. Generates mood content (playlist recommendations)
    4. Draws bounding box on image
    5. Returns base64 image with emotion data
    6. Stores result in session history
    """
    temp_path = None
    try:
        if 'username' not in session:
            return jsonify({'message': 'Not authenticated'}), 401
        
        if 'image' not in request.files:
            return jsonify({'message': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'message': 'Invalid file'}), 400
        
        # Create a unique temp file with explicit control
        import uuid
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # Save the uploaded file to temp location
        file.save(temp_path)
        
        # Recognize emotion
        emotion_result = recognize_emotion(temp_path)
        
        if emotion_result is None:
            return jsonify({'message': 'No face detected in image'}), 400
        
        # Generate mood content (AI-generated playlist recommendation)
        mood_text = generate_mood_content(emotion_result['label'], emotion_result['confidence'])
        
        # Read image and draw bounding box
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({'message': 'Error reading image'}), 500
        
        # Draw bounding box around detected face with emotion label
        box = emotion_result['box']
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{emotion_result['label']} ({emotion_result['confidence']:.1f}%)", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert image to base64 for transmission to frontend
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Generate timestamp-based filename for history tracking
        filename = f"{emotion_result['label']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert confidence to regular float for JSON serialization
        confidence = float(emotion_result['confidence'])
        
        # Store emotion data in session history for dashboard analytics
        if 'emotion_history' not in session:
            session['emotion_history'] = []
        
        history_entry = {
            'emotion': emotion_result['label'],
            'confidence': confidence,
            'filename': filename
        }
        session['emotion_history'].append(history_entry)
        session.modified = True
        
        # Return processed image and analysis data to frontend
        return jsonify({
            'image': f'data:image/jpeg;base64,{img_base64}',
            'emotion': emotion_result['label'],
            'confidence': confidence,
            'mood_text': mood_text
        })
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500
    finally:
        # Clean up temporary file to prevent disk space waste
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp file {temp_path}: {e}")

@app.route('/api/get-history')
def get_history():
    """Retrieve emotion detection history for current session"""
    if 'username' not in session:
        return jsonify({'message': 'Not authenticated'}), 401
    
    history = session.get('emotion_history', [])
    return jsonify({'history': history})

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear all emotion history for current session"""
    if 'username' not in session:
        return jsonify({'message': 'Not authenticated'}), 401
    
    session['emotion_history'] = []
    session.modified = True
    return jsonify({'message': 'History cleared'})

if __name__ == '__main__':
    # Run Flask development server
    app.run(debug=True)
