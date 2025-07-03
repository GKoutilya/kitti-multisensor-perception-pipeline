import os
import cv2
import matplotlib.pyplot as plt

# Path to your dataset folder — update this to your actual path
dataset_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26_drive_0001_sync'

# Path to left camera images
image_folder = os.path.join(dataset_path, 'image_02', 'data')

# List image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
print(f"Number of images found: {len(image_files)}")

# Load the first image
first_image_path = os.path.join(image_folder, image_files[0])
image = cv2.imread(first_image_path)

# Convert BGR (OpenCV default) to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(image_rgb)
plt.title('First KITTI Left Camera Image')
plt.axis('off')
plt.show()
