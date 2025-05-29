import os
import gdown
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

def download_pretrained_model():
    """
    Download a pre-trained skin cancer classification model.
    This is an alternative to training the model from scratch.
    """
    print("Downloading pre-trained skin cancer model...")
    
    # URL for the pre-trained model (this is a placeholder - you would replace with your actual hosted model)
    # For a real project, you would host this model on a reliable file hosting service
    model_url = "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID"
    
    output_path = 'models/skin_cancer_model.h5'
    
    try:
        # Download the model
        gdown.download(model_url, output_path, quiet=False)
        print(f"Model downloaded successfully to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Creating a basic model instead...")
        return False

def create_basic_model():
    """
    Create a basic ResNet50 model for skin cancer classification.
    This is used if downloading the pre-trained model fails.
    """
    print("Creating a basic ResNet50 model...")
    
    # Number of classes in the skin cancer dataset
    NUM_CLASSES = 7
    
    # Load pre-trained ResNet50 without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Save the model
    model.save('models/skin_cancer_model.h5')
    print("Basic model created and saved as 'models/skin_cancer_model.h5'")
    print("Note: This is an untrained model. For accurate predictions, you should train it using train_model.py")

def main():
    """Main function to download or create the model."""
    # Check if model already exists
    if os.path.exists('models/skin_cancer_model.h5'):
        print("Model already exists at 'models/skin_cancer_model.h5'")
        return
    
    # Try to download pre-trained model
    success = download_pretrained_model()
    
    # If download fails, create a basic model
    if not success:
        create_basic_model()

if __name__ == "__main__":
    main()

