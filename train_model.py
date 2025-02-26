import os
import shutil
import kagglehub
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Set paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
KAGGLE_DATASET_PATH = os.path.expanduser("~/.kaggle")

def configure_kaggle_api():
    custom_kaggle_path = os.path.join(PROJECT_DIR, "kaggle.json")
    if not os.path.exists(KAGGLE_DATASET_PATH):
        os.makedirs(KAGGLE_DATASET_PATH)
    shutil.copy(custom_kaggle_path, os.path.join(KAGGLE_DATASET_PATH, "kaggle.json"))
    os.chmod(os.path.join(KAGGLE_DATASET_PATH, "kaggle.json"), 0o600)
    print("✅ Kaggle API key configured successfully!")

def download_and_prepare_dataset():
    print("📥 Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
    print("✅ Dataset downloaded successfully!")

    # Move dataset to the project’s data directory
    target_path = os.path.join(DATA_DIR, "animal-image-dataset")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    shutil.move(dataset_path, target_path)

    # Find the directory containing the image subfolders
    image_base_path = find_image_directory(target_path)
    print(f"✅ Found image base directory: {image_base_path}")
    return image_base_path

def find_image_directory(base_dir):
    """
    Search recursively to find the directory containing the class subfolders (e.g., cats, dogs).
    """
    for root, dirs, files in os.walk(base_dir):
        # If we find subdirectories, assume this is the correct path
        if len(dirs) > 1:
            return root
    raise FileNotFoundError("❌ Could not find the directory containing image subfolders.")

def load_data(data_dir, img_size=(128, 128)):
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            
            # Skip files that cannot be read as images
            if img is None:
                print(f"⚠️ Skipping invalid image file: {img_path}")
                continue
            
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)

    images = np.array(images) / 255.0  # Normalize
    labels = to_categorical(np.array(labels), num_classes=len(class_names))
    return images, labels, class_names


def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), #Extracts features from images, Keeps only positive values
        MaxPooling2D(pool_size=(2, 2)),                                 #Reduces size to prevent overfitting
        Conv2D(64, (3, 3), activation='relu'),                          #Learns more complex patterns
        MaxPooling2D(pool_size=(2, 2)),                                 #Further reduces size
        Flatten(),                                                      #Converts 2D matrix to 1D vector
        Dense(128, activation='relu'),                                  #Fully connected layer for learning relationships 128 neurons
        Dropout(0.5),                                                   #Prevents overfitting
        Dense(num_classes, activation='softmax')                        #Predicts probabilities for each class
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Step 1: Configure Kaggle API key
    configure_kaggle_api()

    # Step 2: Download and move dataset to the correct location
    train_dir = download_and_prepare_dataset()

    # Step 3: Load and preprocess the data
    X, y, class_names = load_data(train_dir)

    # Step 4: Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train the model
    model = create_model(input_shape=(128, 128, 3), num_classes=len(class_names))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32) 

    # Step 6: Save the trained model
    model_save_path = os.path.join(PROJECT_DIR, "models", "animal_model.h5")
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    model.save(model_save_path)
    print(f"✅ Model trained and saved at {model_save_path}!")
