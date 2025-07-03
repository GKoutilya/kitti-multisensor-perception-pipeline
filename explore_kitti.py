import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_calib_file(filepath):
    """
    Parses a KITTI calibration file into a dictionary.

    Args:
        filepath (str): Path to calibration file.

    Returns:
        dict: Mapping from key to NumPy array of floats.
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    data[key.strip()] = np.array([float(x) for x in value.strip().split()])
                except ValueError:
                    # Skip lines that can't be parsed as floats
                    continue
    return data

# === Dataset Paths ===
dataset_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26_drive_0001_sync'
calib_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26'

image_folder = os.path.join(dataset_path, 'image_02', 'data')
lidar_folder = os.path.join(dataset_path, 'velodyne_points', 'data')

# === Load and Display First Image ===
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
first_image_path = os.path.join(image_folder, image_files[0])
image = cv2.imread(first_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title('First KITTI Left Camera Image')
plt.axis('off')
plt.show()

# === Load and Display LIDAR Point Cloud ===
lidar_files = sorted([f for f in os.listdir(lidar_folder) if f.endswith('.bin')])
first_lidar_path = os.path.join(lidar_folder, lidar_files[0])
scan = np.fromfile(first_lidar_path, dtype=np.float32).reshape(-1, 4)  # [x, y, z, intensity]

# Plot LIDAR in 3D using matplotlib
def plot_lidar_3d(x, y, z):
    """
    Displays a 3D scatter plot of the LIDAR point cloud.

    Args:
        x (np.array): X coordinates (forward).
        y (np.array): Y coordinates (left).
        z (np.array): Z coordinates (up).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[::10], y[::10], z[::10], c=z[::10], cmap='jet', s=0.3)
    ax.set_title('KITTI LIDAR Point Cloud (Frame 0)')
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Left)')
    ax.set_zlabel('Z (Up)')
    plt.show()

x, y, z = scan[:, 0], scan[:, 1], scan[:, 2]
plot_lidar_3d(x, y, z)

# === Load Calibration Data ===
velo_calib = read_calib_file(os.path.join(calib_path, 'calib_velo_to_cam.txt'))
cam_calib  = read_calib_file(os.path.join(calib_path, 'calib_cam_to_cam.txt'))

# Extract transformation and projection matrices
P2 = cam_calib['P_rect_02'].reshape(3, 4)  # Projection matrix for camera 2
R_rect = np.eye(4)
R_rect[:3, :3] = cam_calib['R_rect_00'].reshape(3, 3)  # Rectification matrix

Tr_velo_to_cam = np.eye(4)
R = velo_calib['R'].reshape(3, 3)
T = velo_calib['T'].reshape(3, 1)

Tr_velo_to_cam = np.eye(4)
Tr_velo_to_cam[:3, :3] = R
Tr_velo_to_cam[:3, 3] = T.flatten()

# === Project LIDAR onto Camera Image ===
def project_lidar_to_image(scan, image_rgb, P2, R_rect, Tr_velo_to_cam):
    """
    Projects 3D LIDAR points onto the 2D camera image plane and overlays them.

    Args:
        scan (np.array): LIDAR scan of shape [N x 4] (x, y, z, reflectance).
        image_rgb (np.array): Camera image in RGB format.
        P2 (np.array): Camera projection matrix [3 x 4].
        R_rect (np.array): Rectification matrix [4 x 4].
        Tr_velo_to_cam (np.array): LIDAR to camera transform [4 x 4].

    Returns:
        np.array: RGB image with projected LIDAR points overlaid.
    """
    # Filter LIDAR points in front of camera (x > 0)
    lidar_hom = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1)))).T  # [4 x N]
    lidar_hom = lidar_hom[:, lidar_hom[0, :] > 0]

    # Transform LIDAR points to rectified camera coordinates
    lidar_cam = R_rect @ Tr_velo_to_cam @ lidar_hom  # [4 x N]

    # Project to image plane
    pts_2d = P2 @ lidar_cam  # [3 x N]
    pts_2d /= pts_2d[2, :]   # Normalize by depth (z)

    # Draw points on image
    projected_img = image_rgb.copy()
    for i in range(pts_2d.shape[1]):
        u, v = int(pts_2d[0, i]), int(pts_2d[1, i])
        if 0 <= u < projected_img.shape[1] and 0 <= v < projected_img.shape[0]:
            cv2.circle(projected_img, (u, v), radius=1, color=(255, 0, 0), thickness=-1)
    return projected_img

projected_img = project_lidar_to_image(scan, image_rgb, P2, R_rect, Tr_velo_to_cam)

# Display result
plt.figure(figsize=(10, 6))
plt.imshow(projected_img)
plt.title('LIDAR Points Projected onto Camera Image')
plt.axis('off')
plt.show()