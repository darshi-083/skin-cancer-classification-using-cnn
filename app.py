import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template, send_from_directory
import uuid
# Set the Matplotlib backend to 'Agg' before importing matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import time

app = Flask(__name__, static_folder='static')

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load the model
MODEL_PATH = 'models/skin_cancer_model.h5'

# Class names
CLASS_NAMES = [
    'Actinic Keratoses (akiec)',
    'Basal Cell Carcinoma (bcc)',
    'Benign Keratosis (bkl)',
    'Dermatofibroma (df)',
    'Melanoma (mel)',
    'Melanocytic Nevi (nv)',
    'Vascular Lesions (vasc)'
]

# Class descriptions
CLASS_DESCRIPTIONS = {
    'Actinic Keratoses (akiec)': 'A precancerous growth caused by sun damage.',
    'Basal Cell Carcinoma (bcc)': 'The most common type of cancer, usually treatable.',
    'Benign Keratosis (bkl)': 'A non-cancerous growth that appears as a waxy brown, black or tan growth.',
    'Dermatofibroma (df)': 'A common benign skin growth that is usually small and firm.',
    'Melanoma (mel)': 'The most serious type of skin cancer that can spread to other parts of the body.',
    'Melanocytic Nevi (nv)': 'Common moles, usually harmless growths on the skin.',
    'Vascular Lesions (vasc)': 'Abnormalities of blood vessels, including hemangiomas and port-wine stains.'
}

# Risk levels
RISK_LEVELS = {
    'Actinic Keratoses (akiec)': 'Moderate',
    'Basal Cell Carcinoma (bcc)': 'High',
    'Benign Keratosis (bkl)': 'Low',
    'Dermatofibroma (df)': 'Low',
    'Melanoma (mel)': 'Very High',
    'Melanocytic Nevi (nv)': 'Low',
    'Vascular Lesions (vasc)': 'Low'
}

# Load model function
def load_keras_model():
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global model variable
model = None

# Preprocess image function
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# Generate visualization function
def generate_visualization(predictions, class_names, img_path, result_path):
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot image
    img = plt.imread(img_path)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot predictions
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, predictions[0], align='center')
    plt.yticks(y_pos, class_names)
    plt.xlabel('Probability')
    plt.title('Prediction Probabilities')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(result_path)
    plt.close('all')  # Make sure to close all figures
    
    # Return base64 encoded image
    with open(result_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    # Load model if not already loaded
    if model is None:
        model = load_keras_model()
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
    
    # Check if files were uploaded
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    
    # Check number of files
    if len(files) < 1 or len(files) > 200:
        return jsonify({'error': 'Please upload between 1 and 200 images'}), 400
    
    results = []
    
    # Process each file
    for file in files:
        if file.filename == '':
            continue
        
        # Generate unique filename
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save file
        file.save(file_path)
        
        try:
            # Preprocess image
            processed_image = preprocess_image(file_path)
            
            # Make prediction
            start_time = time.time()
            predictions = model.predict(processed_image)
            end_time = time.time()
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Generate visualization
            result_filename = f"result_{filename}"
            result_path = os.path.join(RESULTS_FOLDER, result_filename)
            viz_base64 = generate_visualization(predictions, CLASS_NAMES, file_path, result_path)
            
            # Add result
            results.append({
                'filename': filename,
                'original_filename': file.filename,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'description': CLASS_DESCRIPTIONS[predicted_class],
                'risk_level': RISK_LEVELS[predicted_class],
                'processing_time': round(end_time - start_time, 3),
                'visualization': viz_base64,
                'result_filename': result_filename
            })
            
        except Exception as e:
            results.append({
                'filename': filename,
                'original_filename': file.filename,
                'error': str(e)
            })
    
    return jsonify({
        'results': results,
        'total_processed': len(results)
    })

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    query = data.get('query', '')
    context = data.get('context', {})
    
    # Simple rule-based chatbot responses
    response = get_chatbot_response(query, context)
    
    return jsonify({
        'response': response
    })

def get_chatbot_response(query, context):
    """Generate a response based on the user's query and context."""
    query = query.lower()
    
    # If we have diagnosis context, use it for more specific responses
    diagnosis = context.get('diagnosis', '')
    risk_level = context.get('risk_level', '')
    
    # General diagnosis explanation
    if "what does this diagnosis mean" in query or "what is" in query:
        if diagnosis:
            return f"{diagnosis} is {CLASS_DESCRIPTIONS.get(diagnosis, 'a type of skin lesion')}. " + \
                   get_condition_details(diagnosis)
        else:
            return "I can explain what different skin conditions mean. If you're looking at a specific diagnosis, please ask about that condition by name."
    
    # Should I see a doctor
    elif "should i see a doctor" in query or "medical advice" in query:
        if risk_level in ["High", "Very High"]:
            return f"Yes, you should consult with a dermatologist as soon as possible. This analysis indicates a {risk_level.lower()} risk level, which warrants professional medical attention."
        elif risk_level == "Moderate":
            return "It's recommended to have this checked by a dermatologist. While not an emergency, moderate risk conditions should be evaluated by a professional."
        elif risk_level == "Low":
            return "While this appears to be a low-risk condition, it's always a good idea to have any skin concerns checked by a healthcare professional during your regular check-ups."
        else:
            return "As a general rule, any changes in skin appearance, especially moles or lesions that change in size, shape, or color, should be evaluated by a dermatologist."
    
    # Symptoms
    elif "symptoms" in query or "what are the signs" in query:
        if diagnosis:
            return get_symptoms(diagnosis)
        else:
            return "Different skin conditions have different symptoms. Could you specify which condition you're asking about?"
    
    # Accuracy
    elif "accurate" in query or "reliability" in query:
        return "This AI system uses an ResNet50 model trained on thousands of dermatological images. While it achieves approximately 85-90% accuracy on test datasets, it should not replace professional medical diagnosis. Always consult with a healthcare provider for proper evaluation and treatment."
    
    # Treatment options
    elif "treatment" in query or "how to treat" in query:
        if diagnosis:
            return get_treatment_info(diagnosis)
        else:
            return "Treatment options vary widely depending on the specific condition. Could you specify which condition you're asking about?"
    
    # Prevention
    elif "prevent" in query or "prevention" in query:
        return "General skin cancer prevention includes: limiting sun exposure, using broad-spectrum sunscreen (SPF 30+), wearing protective clothing, avoiding tanning beds, performing regular skin self-exams, and seeing a dermatologist annually if you have risk factors."
    
    # Default response
    else:
        return "I can help answer questions about skin conditions, their symptoms, risk levels, and when to see a doctor. What would you like to know?"

def get_condition_details(diagnosis):
    """Get detailed information about a specific condition."""
    details = {
        'Actinic Keratoses (akiec)': "Actinic keratoses are rough, scaly patches that develop from years of sun exposure. They're considered precancerous because they can develop into skin cancer if left untreated.",
        
        'Basal Cell Carcinoma (bcc)': "Basal cell carcinoma is the most common form of skin cancer. It typically appears as a pearly or waxy bump, a flat, flesh-colored or brown scar-like lesion, or a bleeding or scabbing sore that heals and returns.",
        
        'Benign Keratosis (bkl)': "Benign keratoses are non-cancerous skin growths that appear as waxy, stuck-on-the-skin growths. They're very common after age 40 and are harmless, though they can sometimes be irritated by clothing or jewelry.",
        
        'Dermatofibroma (df)': "Dermatofibromas are common benign skin growths that often appear as small, firm bumps on the legs. They're usually painless and may be pink, gray, red, or brown in color.",
        
        'Melanoma (mel)': "Melanoma is the most serious form of skin cancer. It develops in the cells that produce melanin, the pigment that gives skin its color. Melanomas often resemble moles and some develop from moles. They can appear anywhere on the body.",
        
        'Melanocytic Nevi (nv)': "Melanocytic nevi are common moles. They're usually brown, tan, or flesh-colored spots or bumps on the skin that form when melanocytes grow in clusters. Most people have between 10-40 moles, and they're usually harmless.",
        
        'Vascular Lesions (vasc)': "Vascular lesions are relatively common abnormalities of the skin and underlying tissues, affecting up to 10% of newborns. They're made up of blood vessels that have developed abnormally, and include hemangiomas, port-wine stains, and other vascular malformations."
    }
    
    return details.get(diagnosis, "No detailed information available for this condition.")

def get_symptoms(diagnosis):
    """Get symptoms for a specific condition."""
    symptoms = {
        'Actinic Keratoses (akiec)': "Symptoms include rough, dry, scaly patches of skin that are usually less than 1 inch in diameter. They can be pink, red, or brown, and may feel itchy, burning, or painful when exposed to sunlight.",
        
        'Basal Cell Carcinoma (bcc)': "Look for a pearly or waxy bump, a flat, flesh-colored or brown scar-like lesion, or a bleeding or scabbing sore that heals and returns. They commonly appear on sun-exposed areas like the face and neck.",
        
        'Benign Keratosis (bkl)': "These appear as waxy, stuck-on-the-skin growths that can be flesh-colored, brown, or black. They have a warty, scaly texture and a 'pasted on' appearance.",
        
        'Dermatofibroma (df)': "These appear as small, firm bumps that are usually pink, gray, red or brown. When pinched, they typically dimple inward (known as the 'dimple sign').",
        
        'Melanoma (mel)': "Warning signs include changes in an existing mole or the development of a new, unusual-looking growth. Look for the ABCDE signs: Asymmetry, Border irregularity, Color variations, Diameter larger than 6mm, and Evolving size, shape, or color.",
        
        'Melanocytic Nevi (nv)': "Common moles are usually round or oval, with a smooth edge and even color (usually pink, tan, or brown). They're typically smaller than 1/4 inch in diameter.",
        
        'Vascular Lesions (vasc)': "These appear as red or purple discolorations or growths on the skin. They may be flat or raised and can vary in size from tiny dots to large patches."
    }
    
    return symptoms.get(diagnosis, "No symptom information available for this condition.")

def get_treatment_info(diagnosis):
    """Get treatment information for a specific condition."""
    treatments = {
        'Actinic Keratoses (akiec)': "Treatment options include cryotherapy (freezing), topical medications (5-fluorouracil, imiquimod), photodynamic therapy, curettage, or surgical removal. Regular follow-ups are important as new lesions may develop.",
        
        'Basal Cell Carcinoma (bcc)': "Treatment typically involves surgical removal, which may include Mohs surgery for high-risk areas. Other options include radiation therapy, cryotherapy, topical medications, or photodynamic therapy depending on the size, location, and depth of the cancer.",
        
        'Benign Keratosis (bkl)': "Since these are benign, treatment is usually not necessary unless they become irritated or for cosmetic reasons. Removal options include cryotherapy, curettage, electrosurgery, or shave excision.",
        
        'Dermatofibroma (df)': "Treatment is usually not necessary as they're benign. If desired for cosmetic reasons or if they cause discomfort, they can be surgically removed.",
        
        'Melanoma (mel)': "Treatment depends on the stage but typically involves surgical removal with a margin of healthy skin. Advanced cases may require lymph node biopsy, immunotherapy, targeted therapy, chemotherapy, or radiation therapy.",
        
        'Melanocytic Nevi (nv)': "Most common moles don't require treatment. However, if a mole changes in appearance or is suspicious, it should be evaluated and possibly removed for biopsy.",
        
        'Vascular Lesions (vasc)': "Treatment options include laser therapy, sclerotherapy (injection of a solution to collapse the blood vessels), surgical removal, or compression therapy depending on the type and location of the lesion."
    }
    
    return treatments.get(diagnosis, "No treatment information available for this condition.")

if __name__ == '__main__':
    app.run(debug=True)