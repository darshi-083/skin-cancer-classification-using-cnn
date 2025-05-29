import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 7
DATASET_PATH = 'dataset/HAM10000'

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def download_dataset():
    """
    Download and extract the HAM10000 dataset.
    You can download it manually from:
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
    """
    print("Please download the HAM10000 dataset from:")
    print("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
    print("Extract it to the 'dataset/HAM10000' directory")
    print("The dataset should contain:")
    print("- HAM10000_metadata.csv")
    print("- HAM10000_images_part1.zip (extract to 'dataset/HAM10000/images')")
    print("- HAM10000_images_part2.zip (extract to 'dataset/HAM10000/images')")

def prepare_data():
    """Prepare the dataset for training."""
    # Read metadata
    df = pd.read_csv(f'{DATASET_PATH}/HAM10000_metadata.csv')
    
    # Map diagnosis to numerical labels (for reference and evaluation)
    diagnosis_mapping = {
        'akiec': 0,  # Actinic Keratoses
        'bcc': 1,    # Basal Cell Carcinoma
        'bkl': 2,    # Benign Keratosis
        'df': 3,     # Dermatofibroma
        'mel': 4,    # Melanoma
        'nv': 5,     # Melanocytic Nevi
        'vasc': 6    # Vascular Lesions
    }
    
    # Store numerical labels for reference
    df['label_num'] = df['dx'].map(diagnosis_mapping)
    # Use string labels for training (required for categorical class_mode)
    df['label'] = df['dx']
    
    # Create path column
    df['path'] = df['image_id'].apply(lambda x: f'{DATASET_PATH}/images/{x}.jpg')
    
    # Check if files exist
    df = df[df['path'].apply(os.path.exists)]
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['label'])
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='dx', data=df)
    plt.title('Class Distribution in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png')
    
    return train_df, val_df, test_df, diagnosis_mapping

def create_data_generators(train_df, val_df, test_df):
    """Create data generators for training, validation, and testing."""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and testing
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def build_model():
    """Build and compile the ResNet50 model."""
    # Load pre-trained ResNet50 without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
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
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model(model, train_generator, val_generator):
    """Train the model with callbacks."""
    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/model_checkpoint.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history

def fine_tune_model(model, base_model, train_generator, val_generator):
    """Fine-tune the model by unfreezing some layers."""
    # Unfreeze the last 30 layers of the base model
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune the model
    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[
            ModelCheckpoint('models/fine_tuned_model.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
        ]
    )
    
    return fine_tune_history

def evaluate_model(model, test_generator, diagnosis_mapping):
    """Evaluate the model on the test set."""
    # Get the class indices mapping from the generator
    class_indices = test_generator.class_indices
    
    # Predict on test data
    test_generator.reset()
    y_pred = model.predict(test_generator, steps=len(test_generator), verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Get class names from the generator's class indices
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    
    # Save the final model
    model.save('models/skin_cancer_model.h5')
    print("Model saved as 'models/skin_cancer_model.h5'")

def plot_training_history(history, fine_tune_history=None):
    """Plot training and validation accuracy/loss."""
    # Combine histories if fine-tuning was done
    if fine_tune_history:
        acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
        val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
        loss = history.history['loss'] + fine_tune_history.history['loss']
        val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']
        epochs_range = range(1, len(acc) + 1)
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(acc) + 1)
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')

def main():
    """Main function to run the training pipeline."""
    # Check if dataset exists, otherwise provide download instructions
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH, exist_ok=True)
        download_dataset()
        return
    
    # Prepare data
    train_df, val_df, test_df, diagnosis_mapping = prepare_data()
    
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators(train_df, val_df, test_df)
    
    # Print class indices for reference
    print("Class indices:", train_generator.class_indices)
    
    # Build model
    model, base_model = build_model()
    print(model.summary())
    
    # Train model
    history = train_model(model, train_generator, val_generator)
    
    # Fine-tune model
    fine_tune_history = fine_tune_model(model, base_model, train_generator, val_generator)
    
    # Plot training history
    plot_training_history(history, fine_tune_history)
    
    # Evaluate model
    evaluate_model(model, test_generator, diagnosis_mapping)
    
    print("Training complete! The model is saved as 'models/skin_cancer_model.h5'")

if __name__ == "__main__":
    main()