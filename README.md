# Skin Cancer Classification Project

An advanced skin cancer classification system using ResNet50 with a web interface for image analysis.

## Features

- **Deep Learning Model**: Uses ResNet50 pre-trained on ImageNet and fine-tuned on skin cancer images
- **Multi-class Classification**: Identifies 7 different types of skin lesions
- **Batch Processing**: Analyze up to 200 images at once
- **Interactive UI**: User-friendly web interface with detailed results
- **Visualization**: Visual representation of model predictions and confidence levels

## Skin Lesion Classes

1. Actinic Keratoses (akiec)
2. Basal Cell Carcinoma (bcc)
3. Benign Keratosis (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic Nevi (nv)
7. Vascular Lesions (vasc)

## Project Structure
- `train_model.py`: Script to train the ResNet50 model on the HAM10000 dataset
- `app.py`: Flask web application for serving the model and UI
- `download_model.py`: Script to download a pre-trained model or create a basic one
- `templates/`: HTML templates for the web interface
- `static/`: CSS, JavaScript, and image files
- `models/`: Directory to store the trained model
- `uploads/`: Directory for uploaded images
- `results/`: Directory for visualization results

## Setup and Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the HAM10000 dataset or use your own dataset
4. Train the model: `python train_model.py`
5. Run the application: `python app.py`

## Usage

1. Open the web application in your browser
2. Upload one or more skin lesion images (up to 200)
3. Click "Analyze Images" to process the images
4. View the results with diagnosis, confidence level, and risk assessment
5. Click on any result to see detailed information and visualization

## Performance

The ResNet50 model achieves approximately 85-90% accuracy on the HAM10000 test dataset. Processing time varies depending on hardware, but typically ranges from 0.1-0.5 seconds per image on modern CPUs, and faster on GPUs.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper evaluation of skin lesions.

