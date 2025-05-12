import os
import cv2  # OpenCV for preprocessing
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib

# Load the pre-trained VGG16 model for feature extraction
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


def load_and_preprocess_image(image_path):
    """Load and preprocess an image using OpenCV."""
    image = cv2.imread(image_path)  # Load the image using OpenCV
    if image is None:
        print(f"Warning: Unable to load image {image_path}")
        return None
    image = cv2.resize(image, (224, 224))  # Resize the image using OpenCV
    image = img_to_array(image)  # Convert the image to a NumPy array
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    image = preprocess_input(image)  # Preprocess the image for VGG16
    return image


def extract_features_from_directory(directory, label):
    """Extract features and labels from a directory of images."""
    features = []
    labels = []
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return features, labels

    filenames = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    for filename in tqdm(filenames, desc=f"Processing {directory}"):
        image_path = os.path.join(directory, filename)
        preprocessed_image = load_and_preprocess_image(image_path)
        if preprocessed_image is not None:
            features.append(vgg16_model.predict(preprocessed_image).flatten())
            labels.append(label)
    print(f"Extracted {len(features)} features from {directory}")
    return features, labels


# Specify the paths to your dataset
fire_images_path = r'C:\Users\eshwa\Downloads\detect_fire\fire_dataset\fire_images2'  # Change this path to your fire images directory
non_fire_images_path = r'C:\Users\eshwa\Downloads\detect_fire\fire_dataset\non_fire_images2'  # Change this path to your non-fire images directory

# Load and preprocess fire and non-fire images
fire_features, fire_labels = extract_features_from_directory(fire_images_path, 1)
non_fire_features, non_fire_labels = extract_features_from_directory(non_fire_images_path, 0)

# Combine features and labels
print("Combining features and labels...")
X = np.array(fire_features + non_fire_features)
y = np.array(fire_labels + non_fire_labels)

# Check if there are any samples
print(f"Total samples: {len(X)}")
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

if len(X) == 0:
    print("No samples found. Please check your dataset paths and ensure the directories contain .png images.")
else:
    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM model
    print("Training the SVM model...")
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred))
