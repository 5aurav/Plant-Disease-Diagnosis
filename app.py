from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import uuid
import sys
from pathlib import Path

# Add the project directory to the path so we can import the model
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

# Import the classifier
from model.inference import PotatoDiseaseClassifier

app = Flask(__name__)

classifier = None

# Initialize classifier
try:
    classifier = PotatoDiseaseClassifier()
    print("Classifier initialized successfully")
except Exception as e:
    print(f"Error initializing classifier: {e}")
    # Continue anyway, we'll handle the error if predict is called

# Configure upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'})
    """Process image and return prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Generate unique filename
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save file
    file.save(file_path)
    print(f"Saved image to: {file_path}")
    
    # Make prediction
    try:
        result = classifier.predict(file_path)
        
        # Create response with image path for display
        response = {
            'success': True,
            'prediction': result['class'],
            'confidence': f"{result['confidence'] * 100:.2f}%",
            'image_path': f"static/uploads/{filename}",  # Updated path for client-side use
            'class_probabilities': result['class_probabilities']
        }
        
        return jsonify(response)
    except Exception as e:
        import traceback
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)})
    
@app.route('/get_uploaded_images', methods=['GET'])
def get_uploaded_images():
    """Retrieve list of uploaded images with their predictions"""
    try:
        # Get list of files in the uploads directory
        uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
        
        # Filter out non-image files
        image_files = [f for f in uploaded_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        # Prepare image data with predictions
        image_data = []
        for filename in image_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Try to predict or retrieve previous prediction
            try:
                # If classifier is not loaded, skip this image
                if classifier is None:
                    continue
                
                result = classifier.predict(file_path)
                image_data.append({
                    'path': f'static/uploads/{filename}',
                    'prediction': result['class'],
                    'confidence': result['confidence']
                })
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
        
        return jsonify(image_data)
    
    except Exception as e:
        print(f"Error retrieving uploaded images: {e}")
        return jsonify({'error': 'Could not retrieve uploaded images'}), 500

if __name__ == '__main__':
    app.run(debug=True)