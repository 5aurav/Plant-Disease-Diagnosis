import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore

class PotatoDiseaseClassifier:
    """Class for making predictions on potato leaf images"""
    def __init__(self, model_path="potato_disease_model_final_test.h5", class_indices_path="class_indices.json"):
        # Set default paths relative to the current file
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'potato_disease_model_final.h5')
        if class_indices_path is None:
            class_indices_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                             'class_indices.json')
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        
        # Load class indices
        print(f"Loading class indices from: {class_indices_path}")
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
            
        # Invert the dictionary to map indices to class names
        self.classes = {v: k for k, v in self.class_indices.items()}
        
    def preprocess_image(self, img_path, target_size=(224, 224)):
        """Preprocess image for model input"""
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    
    def predict(self, img_path):
        """Make prediction on image"""
        processed_img = self.preprocess_image(img_path)
        predictions = self.model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_class = self.classes[predicted_class_idx]
        
        result = {
            'class': predicted_class,
            'confidence': confidence,
            'class_probabilities': {self.classes[i]: float(prob) for i, prob in enumerate(predictions[0])}
        }
        
        return result

if __name__ == "__main__":
    # Example usage
    classifier = PotatoDiseaseClassifier()
    result = classifier.predict("path/to/test_image.jpg")
    print(f"Prediction: {result['class']} with confidence {result['confidence']:.2f}")