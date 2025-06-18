from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import numpy as np
from PIL import Image
import io
import requests
from urllib.parse import urlparse
import cv2
import mediapipe as mp
import json
import logging

# Optional ML imports (can be removed if not needed)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available. Outfit recommendations disabled.")

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=True,
    min_detection_confidence=0.5
)

class VirtualTryOnEngine:
    def __init__(self):
        self.pose_detector = pose
        self.clothing_cache = {}
        
    def download_image(self, url):
        """Download image from URL with error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Validate image
            img = Image.open(io.BytesIO(response.content))
            img.verify()  # Verify it's a valid image
            
            # Re-open for processing (verify() closes the image)
            img = Image.open(io.BytesIO(response.content))
            return img
            
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {str(e)}")
            raise ValueError(f"Invalid image URL: {str(e)}")
    
    def validate_image_url(self, url):
        """Validate if URL points to an image"""
        if not url:
            return False
            
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False
                
            # Check file extension
            path = parsed.path.lower()
            return any(path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])
            
        except Exception:
            return False
    
    def detect_pose_landmarks(self, image_array):
        """Detect pose landmarks using MediaPipe"""
        try:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_image)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                return landmarks
            return None
            
        except Exception as e:
            logger.error(f"Pose detection failed: {str(e)}")
            return None
    
    def calculate_clothing_positions(self, landmarks, image_shape):
        """Calculate optimal positions for clothing items based on pose landmarks"""
        if not landmarks or len(landmarks) < 33:
            return None
            
        h, w = image_shape[:2]
        
        # Key landmark indices (MediaPipe pose landmarks)
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        NOSE = 0
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        
        positions = {}
        
        try:
            # Upper wear positioning
            if landmarks[LEFT_SHOULDER]['visibility'] > 0.5 and landmarks[RIGHT_SHOULDER]['visibility'] > 0.5:
                left_shoulder = (int(landmarks[LEFT_SHOULDER]['x'] * w), int(landmarks[LEFT_SHOULDER]['y'] * h))
                right_shoulder = (int(landmarks[RIGHT_SHOULDER]['x'] * w), int(landmarks[RIGHT_SHOULDER]['y'] * h))
                
                shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
                shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2, 
                                 (left_shoulder[1] + right_shoulder[1]) // 2)
                
                positions['upper_wear'] = {
                    'center': shoulder_center,
                    'width': int(shoulder_width * 1.3),
                    'height': int(shoulder_width * 1.0),
                    'angle': 0  # Could calculate based on shoulder slope
                }
            
            # Bottom wear positioning
            if landmarks[LEFT_HIP]['visibility'] > 0.5 and landmarks[RIGHT_HIP]['visibility'] > 0.5:
                left_hip = (int(landmarks[LEFT_HIP]['x'] * w), int(landmarks[LEFT_HIP]['y'] * h))
                right_hip = (int(landmarks[RIGHT_HIP]['x'] * w), int(landmarks[RIGHT_HIP]['y'] * h))
                
                hip_width = abs(right_hip[0] - left_hip[0])
                hip_center = ((left_hip[0] + right_hip[0]) // 2, 
                             (left_hip[1] + right_hip[1]) // 2)
                
                # Calculate pants length to ankles
                if landmarks[LEFT_ANKLE]['visibility'] > 0.5:
                    pants_length = int(landmarks[LEFT_ANKLE]['y'] * h) - hip_center[1]
                else:
                    pants_length = int(hip_width * 2.5)
                
                positions['bottom_wear'] = {
                    'center': hip_center,
                    'width': int(hip_width * 1.4),
                    'height': max(pants_length, int(hip_width * 2)),
                    'angle': 0
                }
            
            # Glasses positioning
            if landmarks[NOSE]['visibility'] > 0.5:
                nose_pos = (int(landmarks[NOSE]['x'] * w), int(landmarks[NOSE]['y'] * h))
                
                positions['glasses'] = {
                    'center': nose_pos,
                    'width': int(w * 0.15),  # Relative to image width
                    'height': int(w * 0.06),
                    'angle': 0
                }
            
            # Shoes positioning
            if landmarks[LEFT_ANKLE]['visibility'] > 0.5 and landmarks[RIGHT_ANKLE]['visibility'] > 0.5:
                left_ankle = (int(landmarks[LEFT_ANKLE]['x'] * w), int(landmarks[LEFT_ANKLE]['y'] * h))
                right_ankle = (int(landmarks[RIGHT_ANKLE]['x'] * w), int(landmarks[RIGHT_ANKLE]['y'] * h))
                
                shoe_width = int(w * 0.12)
                shoe_height = int(w * 0.08)
                
                positions['shoes'] = {
                    'left': {
                        'center': left_ankle,
                        'width': shoe_width,
                        'height': shoe_height,
                        'angle': 0
                    },
                    'right': {
                        'center': right_ankle,
                        'width': shoe_width,
                        'height': shoe_height,
                        'angle': 0
                    }
                }
            
            return positions
            
        except Exception as e:
            logger.error(f"Error calculating clothing positions: {str(e)}")
            return None

# Initialize the try-on engine
tryon_engine = VirtualTryOnEngine()

# Optional: Simple outfit recommendation system
class OutfitRecommender:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.label_encoder = LabelEncoder()
            self._train_model()
    
    def _train_model(self):
        """Train a simple recommendation model with dummy data"""
        # Sample training data (in real app, this would be from user preferences)
        training_data = [
            [1, 1, 0, 1, 'casual'],  # male, shirt, no glasses, sneakers
            [1, 2, 1, 1, 'formal'],  # male, suit, glasses, dress shoes
            [0, 1, 0, 2, 'casual'],  # female, blouse, no glasses, boots
            [0, 3, 1, 1, 'formal'],  # female, dress, glasses, heels
            [1, 1, 0, 2, 'sporty'],  # male, t-shirt, no glasses, boots
            # Add more sample data...
        ]
        
        X = [row[:-1] for row in training_data]
        y = [row[-1] for row in training_data]
        
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        self.model.fit(X, y_encoded)
        self.is_trained = True
    
    def recommend_style(self, gender, upper_type, has_glasses, shoe_type):
        """Recommend outfit style based on selections"""
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return "casual"  # Default recommendation
        
        try:
            gender_code = 1 if gender == 'male' else 0
            glasses_code = 1 if has_glasses else 0
            
            features = [[gender_code, upper_type, glasses_code, shoe_type]]
            prediction = self.model.predict(features)
            style = self.label_encoder.inverse_transform(prediction)[0]
            
            return style
        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            return "casual"

# Initialize recommender
recommender = OutfitRecommender()

@app.route('/')
def index():
    """Serve the main application"""
    return send_from_directory('.', 'index.html')

@app.route('/api/validate-image', methods=['POST'])
def validate_image():
    """Validate image URL and return basic info"""
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'valid': False, 'error': 'No URL provided'})
        
        if not tryon_engine.validate_image_url(url):
            return jsonify({'valid': False, 'error': 'Invalid image URL format'})
        
        # Try to download and validate the image
        try:
            img = tryon_engine.download_image(url)
            return jsonify({
                'valid': True,
                'width': img.width,
                'height': img.height,
                'format': img.format
            })
        except Exception as e:
            return jsonify({'valid': False, 'error': str(e)})
            
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return jsonify({'valid': False, 'error': 'Server error during validation'})

@app.route('/api/detect-pose', methods=['POST'])
def detect_pose():
    """Detect pose landmarks from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)
        
        # Detect pose
        landmarks = tryon_engine.detect_pose_landmarks(img_array)
        
        if landmarks:
            positions = tryon_engine.calculate_clothing_positions(landmarks, img_array.shape)
            return jsonify({
                'success': True,
                'landmarks': landmarks,
                'clothing_positions': positions
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No pose detected in image'
            })
            
    except Exception as e:
        logger.error(f"Pose detection error: {str(e)}")
        return jsonify({'error': 'Pose detection failed'}), 500

@app.route('/api/recommend-outfit', methods=['POST'])
def recommend_outfit():
    """Get outfit recommendations based on user selections"""
    try:
        data = request.get_json()
        
        gender = data.get('gender', 'male')
        clothing_items = data.get('clothing_items', {})
        
        # Simple recommendation logic
        recommendations = {
            'style': 'casual',
            'tips': [],
            'color_suggestions': []
        }
        
        # Use ML model if available
        if SKLEARN_AVAILABLE:
            upper_type = 1 if clothing_items.get('upperWear') else 0
            has_glasses = 1 if clothing_items.get('glasses') else 0
            shoe_type = 1 if clothing_items.get('shoes') else 0
            
            style = recommender.recommend_style(gender, upper_type, has_glasses, shoe_type)
            recommendations['style'] = style
        
        # Add style-specific tips
        if recommendations['style'] == 'formal':
            recommendations['tips'] = [
                'Consider matching belt with shoes',
                'Ensure colors complement each other',
                'Well-fitted clothing creates better appearance'
            ]
        elif recommendations['style'] == 'casual':
            recommendations['tips'] = [
                'Mix and match different textures',
                'Don\'t be afraid to experiment with colors',
                'Comfort is key for casual wear'
            ]
        
        return jsonify(recommendations)
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500

@app.route('/api/process-tryon', methods=['POST'])
def process_tryon():
    """Process virtual try-on request"""
    try:
        data = request.get_json()
        
        user_image = data.get('user_image')  # Base64 encoded
        clothing_items = data.get('clothing_items', {})
        gender = data.get('gender', 'male')
        
        # Validate inputs
        if not user_image and not any(clothing_items.values()):
            return jsonify({'error': 'No user image or clothing items provided'}), 400
        
        # Process clothing items
        processed_items = {}
        for item_type, url in clothing_items.items():
            if url:
                try:
                    if tryon_engine.validate_image_url(url):
                        processed_items[item_type] = {
                            'url': url,
                            'valid': True
                        }
                    else:
                        processed_items[item_type] = {
                            'url': url,
                            'valid': False,
                            'error': 'Invalid URL format'
                        }
                except Exception as e:
                    processed_items[item_type] = {
                        'url': url,
                        'valid': False,
                        'error': str(e)
                    }
        
        # If user image is provided, detect pose
        pose_data = None
        if user_image:
            try:
                # Decode base64 image
                img_data = base64.b64decode(user_image.split(',')[1])
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                
                landmarks = tryon_engine.detect_pose_landmarks(img_array)
                if landmarks:
                    pose_data = tryon_engine.calculate_clothing_positions(landmarks, img_array.shape)
                
            except Exception as e:
                logger.error(f"User image processing failed: {str(e)}")
        
        return jsonify({
            'success': True,
            'processed_items': processed_items,
            'pose_data': pose_data,
            'recommendations': recommender.recommend_style(
                gender, 
                1 if clothing_items.get('upperWear') else 0,
                1 if clothing_items.get('glasses') else 0,
                1 if clothing_items.get('shoes') else 0
            )
        })
        
    except Exception as e:
        logger.error(f"Try-on processing error: {str(e)}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'services': {
            'pose_detection': True,
            'image_processing': True,
            'ml_recommendations': SKLEARN_AVAILABLE
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🎭 Virtual Try-On Server Starting...")
    print("📋 Available endpoints:")
    print("   GET  /                    - Main application")
    print("   POST /api/validate-image  - Validate image URLs")
    print("   POST /api/detect-pose     - Detect pose in images")
    print("   POST /api/recommend-outfit - Get outfit recommendations")
    print("   POST /api/process-tryon   - Process try-on requests")
    print("   GET  /api/health          - Health check")
    print("\n💡 Features:")
    print("   ✅ MediaPipe pose detection")
    print("   ✅ Image validation and processing")
    print("   ✅ Real-time webcam support")
    print("   ✅ Cross-origin resource sharing (CORS)")
    if SKLEARN_AVAILABLE:
        print("   ✅ ML-based outfit recommendations")
    else:
        print("   ⚠️  ML recommendations disabled (install scikit-learn)")
    
    print("\n🚀 Starting server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)