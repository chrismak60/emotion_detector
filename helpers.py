"""
Emotion Detection Helpers
Contains core functions for emotion recognition and mood content generation
"""
from deepface import DeepFace 
import ollama

def recognize_emotion(img_path):
    """
    Detect face and emotion in an image using DeepFace
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        dict: Contains bounding box, emotion label, and confidence score
        Returns None if no face detected
    """
    if img_path is None:
        return None

    # Analyze image using DeepFace with emotion detection
    results = DeepFace.analyze(
        img_path = img_path, 
        actions = ['emotion'], 
        detector_backend = 'retinaface', # Recommended for stable bounding boxes 
        enforce_detection = False
    )
    
    # Extract first detected face data
    face_data = results[0]
    
    return {
        "box": face_data['region'], # Bounding box coordinates (x, y, w, h)
        "label": face_data['dominant_emotion'], # Detected emotion
        "confidence": face_data['emotion'][face_data['dominant_emotion']]# Confidence score
    }

def generate_mood_content(emotion_label, confidence):
    """
    Generate AI-based mood insights and playlist recommendation
    Uses Ollama with Gemma model for fast, lightweight text generation
    
    Args:
        emotion_label (str): The detected emotion
        confidence (float): Confidence score (0-100)
        
    Returns:
        str: AI-generated mood description with music playlist suggestion
    """
    # Create prompt for AI to generate contextual mood content
    prompt = f"The detected emotion is {emotion_label}. Suggest a 3-song playlist that matches it. Start by saying: 'Tonight's vibe is {emotion_label}. I'm {confidence:.1f}% sure about it... (here give a short description of the mood)... For that reason, here are some songs to match the mood.' and then give the playlist you suggest. No final questions or suggestions."
    
    # Call Ollama with Gemma model for music curation
    response = ollama.chat(
        model="gemma3:1b", # Lightweight model for real-time applications
        messages=[
            {"role": "system", "content": "You are a professional music curator and psychologist."},
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0.7  # High temperature for creativity in music suggestions
        }
    )
    
    return response['message']['content']
    
