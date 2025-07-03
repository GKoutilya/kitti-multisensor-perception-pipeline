import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ========== CONFIG ==========
dataset_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26_drive_0001_sync'
calib_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26'

image_folder = os.path.join(dataset_path, 'image_02', 'data')
lidar_folder = os.path.join(dataset_path, 'velodyne_points', 'data')
label_folder = os.path.join(os.path.dirname(__file__), 'label_2')  # Adjust if needed

NUM_FRAMES = 25  # Can increase; watch memory/CPU usage
LIDAR_DOWNSAMPLE = 5
POINT_SIZE = 1.5

def read_calib_file(filepath):
    """
    Parse KITTI calibration file into dictionary of numpy arrays.

    Args:
        filepath (str): Path to calibration file.

    Returns:
        dict: Keys as calibration params, values as numpy arrays.
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    data[key.strip()] = np.array([float(x) for x in value.strip().split()])
                except ValueError:
                    continue
    return data

def load_calibration():
    """
    Load KITTI calibration files and build transformation matrices.

    Returns:
        tuple: (Tr_velo_to_cam, R_rect, P2) as numpy arrays.
    """
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

def project_lidar_to_image(scan, Tr_velo_to_cam, R_rect, P2, color_by='depth'):
    """
    Project LIDAR points to 2D image plane with coloring by depth or intensity.

    Args:
        scan (np.array): Nx4 LIDAR points [x,y,z,intensity].
        Tr_velo_to_cam (np.array): 4x4 LIDAR to camera transform.
        R_rect (np.array): 4x4 rectification matrix.
        P2 (np.array): 3x4 projection matrix.
        color_by (str): 'depth' or 'intensity' coloring.

    Returns:
        tuple:
            - pts_2d (np.array): Nx2 2D pixel coords.
            - colors (np.array): Color values for each point.
    """
    scan = scan[::LIDAR_DOWNSAMPLE]
    scan = scan[scan[:, 0] > 0]  # Keep points in front of vehicle

    lidar_hom = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1))))
    pts_cam = (Tr_velo_to_cam @ lidar_hom.T).T
    pts_rect = (R_rect @ pts_cam.T).T
    pts_rect = pts_rect[pts_rect[:, 2] > 0]  # Keep points with positive depth

    pts_2d = (P2 @ pts_rect.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2, np.newaxis]

    if color_by == 'intensity':
        colors = scan[:, 3]
    else:
        colors = pts_rect[:, 2]  # depth

    return pts_2d, colors

def parse_label_file(label_path):
    """
    Parse KITTI label file for a frame.

    Args:
        label_path (str): Path to label file.

    Returns:
        list of dicts: Each dict has keys like type, truncated, occluded, alpha,
        bbox (2D), dimensions (h,w,l), location (x,y,z), rotation_y.
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
                'bbox': [float(x) for x in parts[4:8]],        # 2D bbox
                'dimensions': [float(x) for x in parts[8:11]], # h,w,l
                'location': [float(x) for x in parts[11:14]],  # x,y,z
                'rotation_y': float(parts[14])
            }
            objects.append(obj)
    return objects

def compute_3d_box(dimensions, location, rotation_y):
    """
    Compute 8 corners of 3D bounding box in camera coordinates.

    Args:
        dimensions (list): [height, width, length]
        location (list): [x, y, z]
        rotation_y (float): Rotation around Y-axis.

    Returns:
        np.array: 8x3 array of corner points.
    """
    h, w, l = dimensions
    x, y, z = location

    # Box corners in object coordinate frame
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.vstack([x_corners, y_corners, z_corners])

    # Rotation matrix around Y axis
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
    Project 3D points to 2D image plane.

    Args:
        pts_3d (np.array): Nx3 points in camera coordinates.
        P (np.array): 3x4 projection matrix.

    Returns:
        np.array: Nx2 array of 2D pixel coordinates.
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = (P @ pts_3d_hom.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2, np.newaxis]
    return pts_2d

def draw_3d_bbox(img, corners, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box on image.

    Args:
        img (np.array): Image to draw on.
        corners (np.array): 8x2 projected corners.
        color (tuple): BGR color.
        thickness (int): Line thickness.
    """
    corners = corners.astype(np.int32)
    # Bottom rectangle
    for i, j in zip([0,1,2,3], [1,2,3,0]):
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)
    # Top rectangle
    for i, j in zip([4,5,6,7], [5,6,7,4]):
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)
    # Vertical lines
    for i, j in zip(range(4), range(4,8)):
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)

class FusionAnimator:
    """
    Animate LIDAR-camera sensor fusion and 3D bounding boxes from KITTI.

    Attributes:
        num_frames (int): Number of frames to animate.
        Tr_velo_to_cam (np.array): LIDAR to camera transform.
        R_rect (np.array): Rectification matrix.
        P2 (np.array): Camera projection matrix.
        image_files (list): Sorted image filenames.
        lidar_files (list): Sorted LIDAR filenames.
        label_files (list): Sorted label filenames.
        fig, ax: Matplotlib figure and axis.
        img_display: Image display object.
        scatter: Scatter plot for LIDAR points.
        paused (bool): Animation pause flag.
        anim: Matplotlib FuncAnimation object.
    """

    def __init__(self, num_frames=NUM_FRAMES):
        self.num_frames = num_frames
        self.Tr_velo_to_cam, self.R_rect, self.P2 = load_calibration()

        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])[:num_frames]
        self.lidar_files = sorted([f for f in os.listdir(lidar_folder) if f.endswith('.bin')])[:num_frames]
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])[:num_frames]

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.img_display = self.ax.imshow(np.zeros((375, 1242, 3), dtype=np.uint8))
        self.scatter = self.ax.scatter([], [], s=POINT_SIZE, c=[], cmap='jet', alpha=0.8)
        self.ax.axis('off')
        self.ax.set_title('KITTI Sensor Fusion with 3D Bounding Boxes')

        axpause = plt.axes([0.81, 0.01, 0.1, 0.05])
        self.btn_pause = Button(axpause, 'Pause')
        self.btn_pause.on_clicked(self.toggle_pause)
        self.paused = False

        self.anim = FuncAnimation(self.fig, self.update, frames=self.num_frames,
                                  interval=100, blit=False)

    def toggle_pause(self, event):
        """Toggle animation pause/play."""
        self.paused = not self.paused
        self.btn_pause.label.set_text('Play' if self.paused else 'Pause')

    def update(self, frame_idx):
        if self.paused:
            return self.img_display, self.scatter

        # Load camera image
        img = cv2.imread(os.path.join(image_folder, self.image_files[frame_idx]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load LIDAR scan and project
        scan = np.fromfile(os.path.join(lidar_folder, self.lidar_files[frame_idx]), dtype=np.float32).reshape(-1, 4)
        pts_2d, colors = project_lidar_to_image(scan, self.Tr_velo_to_cam, self.R_rect, self.P2, color_by='intensity')

        # Load labels and draw 3D bounding boxes **on img_rgb**
        label_path = os.path.join(label_folder, self.label_files[frame_idx])
        objects = parse_label_file(label_path)
        for obj in objects:
            corners_3d = compute_3d_box(obj['dimensions'], obj['location'], obj['rotation_y'])
            corners_2d = project_to_image(corners_3d, self.P2)
            draw_3d_bbox(img_rgb, corners_2d, color=(0, 255, 0), thickness=2)

        # Update image display **after** drawing boxes
        self.img_display.set_data(img_rgb)
        self.scatter.set_offsets(pts_2d)
        self.scatter.set_array(colors)

        return self.img_display, self.scatter


    def save_animation(self, filename='fusion_animation.gif'):
        """
        Save the animation as a GIF file (temporary workaround).

        Args:
            filename (str): Output filename.
        """
        print(f"Saving animation to {filename} ...")
        self.anim.save(filename, writer='pillow', fps=10)
        print("Saved.")


def main():
    """Run the fusion animation."""
    animator = FusionAnimator()
    plt.show()
    animator.save_animation()  # Uncomment to save video

if __name__ == "__main__":
    main()
