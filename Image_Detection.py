# Mount Drive (because the dataset is located in the drive due to its large capacity.) 
from google.colab import drive
drive.mount('/content/drive')

# Import libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.image as mpimg

# Dataset directory path (to make sure that we load the correct data, we check for the dataset directory for subclasses in the ‘Vehicle-detection-data’ folder)
# Define dataset directory paths
dataset_dir = '/content/drive/MyDrive/vehicle-detection-data'
train_dir = dataset_dir
vehicle_dir = os.path.join(dataset_dir, 'vehicles')
non_vehicle_dir = os.path.join(dataset_dir, 'non-vehicles')
# Verify paths for the dataset
print(f"Vehicle images directory: {vehicle_dir}")
print(f"Non-vehicle images directory: {non_vehicle_dir}")

# Data Preprocessing
# Set up ImageDataGenerator for training and validation
#ImageDataGenerator is set up for data augmentation that will be used in training and validation.This is a class from Keras and will be used in the later part of the experiment.
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize images
    shear_range=0.2,  # Random shear transformations
    zoom_range=0.2,   # Random zoom
    horizontal_flip=True  # Random horizontal flip
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Just normalization for validation set

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Resize images to 64x64
    batch_size=32,
    class_mode='binary'  # Binary classification: vehicle vs non-vehicle
)

validation_generator = test_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))

# Define the Convolutional Neural Network (CNN)
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the convolution layers
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Show the model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Evaluate the model on the validation set
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {validation_loss}")
print(f"Validation Accuracy: {validation_accuracy}")

# Function to preprocess and predict a new image
def load_and_predict(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Resize to match input size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return 'Vehicle' if prediction[0] > 0.5 else 'Non-Vehicle'

# Example: Test a single image
from google.colab import files
uploaded = files.upload () # can upload any image file

# Get the uploaded image file name and display it
for filename in uploaded.keys():
    test_image_path = filename  # Store the file path

    # Display the uploaded image
    img = mpimg.imread(test_image_path)  # Read the image file
    plt.imshow(img)  # Display the image
    plt.axis('off')  # Turn off axis
    plt.title(f"Uploaded Image: {filename}")  # Add a title
    plt.show()

# Load and preprocess the image using the defined function
result = load_and_predict(test_image_path)
print(f'Prediction: {result}')

#OPTIMIZATION
def train_model_with_params(train_gen, val_gen, learning_rate, batch_size, epochs=5):
    print(f"\nTraining with Learning Rate: {learning_rate}, Batch Size: {batch_size}")

    # Compile the model with the specified learning rate
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        verbose=1
    )

    # Log the best validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    return best_val_accuracy

def create_generators(batch_size):
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary'
    )

    val_gen = test_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_gen, val_gen

# Experiment with Learning Rates
learning_rates = [0.01, 0.001, 0.0001]
batch_size = 32  # Fixed batch size for learning rate experiments

print("Starting Learning Rate Experiments...")
for lr in learning_rates:
    train_gen, val_gen = create_generators(batch_size)  # Use fixed batch size
    train_model_with_params(train_gen, val_gen, learning_rate=lr, batch_size=batch_size)

# Experiment with Batch Sizes
batch_sizes = [16, 32, 64, 128]
learning_rate = 0.001  # Fixed learning rate for batch size experiments

print("\nStarting Batch Size Experiments...")
for batch_size in batch_sizes:
    train_gen, val_gen = create_generators(batch_size)
    train_model_with_params(train_gen, val_gen, learning_rate=learning_rate, batch_size=batch_size)

print("\nOptimization Experiments Complete.")

