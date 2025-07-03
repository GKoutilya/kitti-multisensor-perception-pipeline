import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from tracker import SimpleTracker
import random

# ========== CONFIG ==========
dataset_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26_drive_0001_sync'
calib_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26'

image_folder = os.path.join(dataset_path, 'image_02', 'data')
lidar_folder = os.path.join(dataset_path, 'velodyne_points', 'data')
label_folder = os.path.join(os.path.dirname(__file__), 'label_2')

NUM_FRAMES = 25
LIDAR_DOWNSAMPLE = 5
POINT_SIZE = 1.5

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
                    continue
    return data

def load_calibration():
    """Load KITTI calibration files and return transformation matrices."""
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
    """Project LIDAR points to 2D image plane with coloring by depth or intensity."""
    scan = scan[::LIDAR_DOWNSAMPLE]
    scan = scan[scan[:, 0] > 0]

    lidar_hom = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1))))
    pts_cam = (Tr_velo_to_cam @ lidar_hom.T).T
    pts_rect = (R_rect @ pts_cam.T).T
    pts_rect = pts_rect[pts_rect[:, 2] > 0]

    pts_2d = (P2 @ pts_rect.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2, np.newaxis]

    colors = scan[:, 3] if color_by == 'intensity' else pts_rect[:, 2]
    return pts_2d, colors

def parse_label_file(label_path):
    """Parse KITTI label file into a list of object dictionaries."""
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
    """Compute 8 corners of 3D bounding box in camera coordinates."""
    h, w, l = dimensions
    x, y, z = location

    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corners = np.vstack([x_corners, y_corners, z_corners])

    R = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    corners_rot = R @ corners
    corners_3d = corners_rot + np.array([[x], [y], [z]])
    return corners_3d.T

def project_to_image(pts_3d, P):
    """Project 3D points to 2D image plane."""
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = (P @ pts_3d_hom.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2, np.newaxis]
    return pts_2d

def draw_3d_bbox(img, corners, color=(0, 255, 0), thickness=2):
    """Draw 3D bounding box on image."""
    corners = corners.astype(np.int32)
    for i, j in zip([0,1,2,3], [1,2,3,0]):
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)
    for i, j in zip([4,5,6,7], [5,6,7,4]):
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)
    for i, j in zip(range(4), range(4,8)):
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)

class FusionAnimator:
    """
    Animate KITTI sensor fusion with 3D bounding boxes and persistent object tracking.

    Tracks object trajectories and displays IDs with unique colors. Saves final trajectory image and animation GIF.
    """
    def __init__(self, num_frames=NUM_FRAMES):
        self.num_frames = num_frames
        self.Tr_velo_to_cam, self.R_rect, self.P2 = load_calibration()
        self.tracker = SimpleTracker(max_distance=3.0)

        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])[:num_frames]
        self.lidar_files = sorted([f for f in os.listdir(lidar_folder) if f.endswith('.bin')])[:num_frames]
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])[:num_frames]

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.img_display = self.ax.imshow(np.zeros((375, 1242, 3), dtype=np.uint8))
        self.scatter = self.ax.scatter([], [], s=POINT_SIZE, c=[], cmap='jet', alpha=0.8)
        self.ax.axis('off')
        self.ax.set_title('KITTI Sensor Fusion with 3D Bounding Boxes & Object Tracking')

        axpause = plt.axes([0.81, 0.01, 0.1, 0.05])
        self.btn_pause = Button(axpause, 'Pause')
        self.btn_pause.on_clicked(self.toggle_pause)
        self.paused = False

        # For tracking colors and trajectories per object ID
        self.colors = {}  # id -> (R,G,B) color tuple in OpenCV format (B,G,R)
        self.trajectories = {}  # id -> list of 2D points (pixel coords) for trajectory lines

        self.anim = FuncAnimation(self.fig, self.update, frames=self.num_frames, interval=100, blit=False)

    def _get_color(self, obj_id):
        """Return a consistent color for a given object ID."""
        if obj_id not in self.colors:
            # Generate a random bright color (BGR) for OpenCV
            self.colors[obj_id] = tuple(np.random.choice(range(50,256), size=3).tolist())
        return self.colors[obj_id]

    def toggle_pause(self, event):
        """Toggle pause/play state of animation."""
        self.paused = not self.paused
        self.btn_pause.label.set_text('Play' if self.paused else 'Pause')

    def update(self, frame_idx):
        """Update animation frame: draw image, lidar, boxes, tracking IDs, and trajectories."""
        if self.paused:
            return self.img_display, self.scatter

        # Load image and convert to RGB
        img = cv2.imread(os.path.join(image_folder, self.image_files[frame_idx]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load lidar scan and project points to image
        scan = np.fromfile(os.path.join(lidar_folder, self.lidar_files[frame_idx]), dtype=np.float32).reshape(-1, 4)
        pts_2d, colors = project_lidar_to_image(scan, self.Tr_velo_to_cam, self.R_rect, self.P2, color_by='intensity')

        # Load labels and parse objects
        label_path = os.path.join(label_folder, self.label_files[frame_idx])
        objects = parse_label_file(label_path)

        # Extract centroids (3D) for tracking
        centroids = [obj['location'] for obj in objects]
        tracked = self.tracker.update(centroids)

        # Draw 3D boxes and tracking info
        for obj, track in zip(objects, tracked):
            corners_3d = compute_3d_box(obj['dimensions'], obj['location'], obj['rotation_y'])
            corners_2d = project_to_image(corners_3d, self.P2)
            color = self._get_color(track['id'])

            # Draw bounding box in RGB image (convert BGR->RGB for matplotlib display)
            draw_3d_bbox(img_rgb, corners_2d, color=color, thickness=2)

            # Project centroid to 2D for text and trajectory
            cx, cy, cz = obj['location']
            pt_2d = project_to_image(np.array([[cx, cy, cz]]), self.P2)[0]
            pt_2d_int = (int(pt_2d[0]), int(pt_2d[1]))

            # Draw ID text near centroid
            cv2.putText(img_rgb, f'ID {track["id"]}', pt_2d_int,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # Update trajectory points for this object ID
            if track['id'] not in self.trajectories:
                self.trajectories[track['id']] = []
            self.trajectories[track['id']].append(pt_2d)

        # Draw all trajectories on img_rgb as lines
        for obj_id, points in self.trajectories.items():
            pts_array = np.array(points, dtype=np.int32)
            # Draw polyline on image (BGR, so color reversed for matplotlib RGB)
            # Convert from RGB to BGR for OpenCV drawing
            color_bgr = tuple(reversed(self.colors[obj_id]))
            for i in range(1, len(pts_array)):
                cv2.line(img_rgb, tuple(pts_array[i-1]), tuple(pts_array[i]), color_bgr, 2)

        # Update matplotlib display
        self.img_display.set_data(img_rgb)
        self.scatter.set_offsets(pts_2d)
        self.scatter.set_array(colors)

        return self.img_display, self.scatter

    def save_trajectory_image(self, filename='trajectory_summary.png'):
        """
        Save a snapshot image of the last frame with all trajectories drawn.

        Args:
            filename (str): Output image filename.
        """
        # Load last frame image
        img = cv2.imread(os.path.join(image_folder, self.image_files[-1]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw trajectories on last frame
        for obj_id, points in self.trajectories.items():
            pts_array = np.array(points, dtype=np.int32)
            color_bgr = tuple(reversed(self.colors[obj_id]))
            for i in range(1, len(pts_array)):
                cv2.line(img_rgb, tuple(pts_array[i-1]), tuple(pts_array[i]), color_bgr, 3)
            # Draw final ID label
            final_pos = pts_array[-1]
            cv2.putText(img_rgb, f'ID {obj_id}', tuple(final_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 3, cv2.LINE_AA)

        # Save using matplotlib (RGB)
        plt.imsave(filename, img_rgb)
        print(f"Trajectory summary image saved as {filename}")

    def save_animation(self, filename='fusion_tracking.gif'):
        """Save the full animation as a GIF file."""
        print(f"Saving animation to {filename} ...")
        self.anim.save(filename, writer='pillow', fps=10)
        print("Saved.")

def main():
    animator = FusionAnimator()
    plt.show()
    animator.save_animation()
    animator.save_trajectory_image()

if __name__ == "__main__":
    main()
