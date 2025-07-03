import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths — update these if needed
dataset_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26_drive_0001_sync'
calib_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26'

image_folder = os.path.join(dataset_path, 'image_02', 'data')
label_folder = os.path.join(os.path.dirname(__file__), 'label_2')  # Adjust if label folder is elsewhere

def read_calib_file(filepath):
    """Parse KITTI calibration file into dictionary of numpy arrays."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    data[key.strip()] = np.array([float(x) for x in value.strip().split()])
                except ValueError:
                    # Some lines like dates or comments might fail, ignore those
                    continue
    return data

def load_calibration():
    """Load necessary calibration matrices."""
    velo_calib = read_calib_file(os.path.join(calib_path, 'calib_velo_to_cam.txt'))
    cam_calib = read_calib_file(os.path.join(calib_path, 'calib_cam_to_cam.txt'))

    R = velo_calib['R'].reshape(3, 3)
    T = velo_calib['T'].reshape(3, 1)

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :3] = R
    Tr_velo_to_cam[:3, 3] = T.flatten()

    R_rect = np.eye(4)
    R_rect[:3, :3] = cam_calib['R_rect_00'].reshape(3, 3)

    P2 = cam_calib['P_rect_02'].reshape(3, 4)

    return Tr_velo_to_cam, R_rect, P2

def parse_label_file(label_path):
    """
    Parse KITTI label file for one frame.

    Args:
        label_path (str): Path to label text file.

    Returns:
        List of dicts with keys: type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y
    """
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            obj = {
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(x) for x in parts[4:8]],
                'dimensions': [float(x) for x in parts[8:11]],
                'location': [float(x) for x in parts[11:14]],
                'rotation_y': float(parts[14])
            }
            objects.append(obj)
    return objects

def compute_3d_box(dimensions, location, rotation_y):
    """
    Compute 8 corners of the 3D bounding box in camera coordinates.

    Args:
        dimensions (list): [height, width, length]
        location (list): [x, y, z]
        rotation_y (float): rotation around Y axis.

    Returns:
        numpy array: 8 corners (8x3)
    """
    h, w, l = dimensions
    x, y, z = location

    # Object coordinate corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.vstack([x_corners, y_corners, z_corners])

    # Rotation matrix
    R = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    corners_rot = R @ corners
    corners_3d = corners_rot + np.array([[x], [y], [z]])

    return corners_3d.T

def project_to_image(pts_3d, P):
    """
    Project 3D points to 2D image.

    Args:
        pts_3d (numpy array): Nx3 points
        P (numpy array): 3x4 projection matrix

    Returns:
        Nx2 array of pixel coordinates
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = (P @ pts_3d_hom.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2, np.newaxis]
    return pts_2d

def draw_3d_bbox(img, corners, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box edges on image.

    Args:
        img (np.array): image
        corners (numpy array): 8x2 corners projected
        color (tuple): BGR color
        thickness (int): line thickness
    """
    corners = corners.astype(np.int32)
    # Bottom rectangle
    for i, j in zip([0,1,2,3], [1,2,3,0]):
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)
    # Top rectangle
    for i, j in zip([4,5,6,7], [5,6,7,4]):
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)
    # Vertical edges
    for i, j in zip(range(4), range(4,8)):
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)

def main():
    Tr_velo_to_cam, R_rect, P2 = load_calibration()

    frame_idx = 0
    image_file = sorted(os.listdir(image_folder))[frame_idx]
    label_file = sorted(os.listdir(label_folder))[frame_idx]

    img_path = os.path.join(image_folder, image_file)
    lbl_path = os.path.join(label_folder, label_file)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    objects = parse_label_file(lbl_path)

    for obj in objects:
        corners_3d = compute_3d_box(obj['dimensions'], obj['location'], obj['rotation_y'])
        corners_2d = project_to_image(corners_3d, P2)
        draw_3d_bbox(img_rgb, corners_2d)

    plt.figure(figsize=(12, 6))
    plt.imshow(img_rgb)
    plt.title('3D Bounding Boxes Projected on KITTI Image')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
