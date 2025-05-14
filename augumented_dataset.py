import os
import shutil
from sklearn.model_selection import train_test_split

# Set paths
original_dataset_dir = 'C:/Users/saura/OneDrive/Desktop/Training/dataset'
base_dir = 'C:/Users/saura/OneDrive/Desktop/Training/plant_dataset'  # final dataset structure will be created here

# Create train and val folders
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Make sure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# For each class, split images
for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    images = os.listdir(class_path)

    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    # Create subfolders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

print("✅ Dataset split into train and validation successfully.")
