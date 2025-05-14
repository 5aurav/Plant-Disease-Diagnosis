import tensorflow as tf
from tensorflow.keras.models import load_model

# Replace this with the path to your .h5 file
model_path = 'potato_disease_model_final.h5'

print(f"Attempting to load model from: {model_path}")

try:
    # Load the model
    model = load_model(model_path)
    
    # Print model summary
    print("\nModel loaded successfully!")
    print("\n=== MODEL SUMMARY ===")
    model.summary()
    
    # Print basic information
    print(f"\nInput shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
except Exception as e:
    print(f"\nError loading model: {str(e)}")
    print("\nPossible issues:")
    print("1. Incorrect file path - Double check that the path to your .h5 file is correct")
    print("2. Missing dependencies - Ensure TensorFlow and all required packages are installed")
    print("3. Custom model components - If your model uses custom layers or metrics, they need to be provided when loading")
    print("\nIf using custom components, try loading with:")
    print("model = load_model(model_path, custom_objects={'CustomLayer': CustomLayer})")