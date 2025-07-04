import os
import cv2
import openai
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from kalman_tracker import MultiObjectTracker

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("my_api_key")

# ========== CONFIGURATION ==========
dataset_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26_drive_0001_sync'
calib_path = r'C:\Users\kouti\Python\(5) Multisensor Fusion & Real-Time 3D Object Tracking Perception Pipeline Useing KITTI Dataset\2011_09_26'

image_folder = os.path.join(dataset_path, 'image_02', 'data')
lidar_folder = os.path.join(dataset_path, 'velodyne_points', 'data')
label_folder = os.path.join(os.path.dirname(__file__), 'label_2')

NUM_FRAMES = 25
LIDAR_DOWNSAMPLE = 5
POINT_SIZE = 1.5

def read_calib_file(filepath):
    """
    Read KITTI calibration text file and parse into dict of numpy arrays.

    Args:
        filepath (str): Path to the KITTI calibration file.

    Returns:
        dict: Keys are parameter names, values are numpy arrays.
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
    Load KITTI calibration matrices needed for sensor fusion.

    Returns:
        tuple: (Tr_velo_to_cam, R_rect, P2) matrices as numpy arrays.
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
    Project 3D LiDAR points into 2D camera image plane with color mapping.

    Args:
        scan (np.ndarray): LiDAR points (N x 4) with intensity in 4th column.
        Tr_velo_to_cam (np.ndarray): 4x4 transform matrix from LiDAR to camera coords.
        R_rect (np.ndarray): 4x4 rectification matrix.
        P2 (np.ndarray): 3x4 camera projection matrix.
        color_by (str): 'depth' or 'intensity' coloring.

    Returns:
        tuple:
            pts_2d (np.ndarray): 2D points projected onto image plane (M x 2).
            colors (np.ndarray): Color values for each point.
    """
    scan = scan[::LIDAR_DOWNSAMPLE]
    scan = scan[scan[:, 0] > 0]  # Remove points behind sensor

    lidar_hom = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1))))  # Homogeneous coords (N x 4)
    pts_cam = (Tr_velo_to_cam @ lidar_hom.T).T  # Transform to camera coords
    pts_rect = (R_rect @ pts_cam.T).T  # Rectify points
    pts_rect = pts_rect[pts_rect[:, 2] > 0]  # Keep points in front of camera

    pts_2d = (P2 @ pts_rect.T).T  # Project to 2D image plane (N x 3)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2, np.newaxis]  # Normalize by depth

    colors = scan[:, 3] if color_by == 'intensity' else pts_rect[:, 2]

    return pts_2d, colors

def parse_label_file(label_path):
    """
    Parse KITTI label file into list of object dicts.

    Args:
        label_path (str): Path to KITTI label txt file.

    Returns:
        list: List of dicts with keys: type, truncated, occluded, alpha, bbox,
              dimensions, location, rotation_y.
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
    Compute 3D bounding box corners in camera coordinates.

    Args:
        dimensions (list): [height, width, length]
        location (list): [x, y, z] location of box center.
        rotation_y (float): Rotation around Y axis in radians.

    Returns:
        np.ndarray: 8 corners of the 3D box (8 x 3).
    """
    h, w, l = dimensions
    x, y, z = location

    # 3D bounding box corners relative to object center
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
    return corners_3d.T  # Shape (8,3)

def project_to_image(pts_3d, P):
    """
    Project 3D points to 2D image plane.

    Args:
        pts_3d (np.ndarray): (N x 3) array of 3D points.
        P (np.ndarray): (3 x 4) camera projection matrix.

    Returns:
        np.ndarray: (N x 2) array of 2D points on image plane.
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = (P @ pts_3d_hom.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2, np.newaxis]
    return pts_2d

def generate_scene_summary(objects, tracked):
    """
    Generate a concise natural language summary of the current traffic scene
    using OpenAI GPT based on detected objects and tracked IDs.

    Args:
        objects (list of dict): Detected objects with types and locations.
        tracked (list): List of tracked objects with Kalman filter state.

    Returns:
        str: Summary text generated by GPT, or error message on failure.
    """
    if not objects or not tracked:
        return "No objects detected in the current frame."

    prompt = "Summarize the current traffic scene in one sentence. Here are the objects:\n"
    for track in tracked:
        track_id = track.track_id
        state = track.get_state()
        if state is None:
            continue
        cx, cy, cz = state[:3]
        closest_obj = min(objects, key=lambda o: np.linalg.norm(np.array(o['location']) - state[:3]))
        obj_type = closest_obj['type']
        prompt += f"- Object ID {track_id}: {obj_type} at position ({cx:.1f}, {cy:.1f}, {cz:.1f})\n"

    prompt += "\nProvide a concise summary like 'Three cars are moving along the road.'"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes traffic scenes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM error: {e})"

class FusionAnimator:
    """
    Class that animates the KITTI sensor fusion pipeline with 3D bounding boxes,
    persistent multi-object tracking via Kalman filter, and GPT-based dynamic scene summaries.

    Features:
    - Left panel: color camera frame overlaid with projected LiDAR points, 3D bounding boxes, and track IDs.
    - Right panel: live natural language summary of scene via OpenAI GPT.
    - Pause/play button below summary.
    - Saves trajectory summary image with object paths and IDs.
    - Saves GIF animation of the sequence.

    Attributes:
        num_frames (int): Number of frames to animate.
        Tr_velo_to_cam (np.ndarray): Velodyne to camera transform matrix.
        R_rect (np.ndarray): Rectification matrix.
        P2 (np.ndarray): Projection matrix for left color camera.
        tracker (MultiObjectTracker): Kalman filter multi-object tracker instance.
        image_files (list): Sorted list of color image filenames.
        lidar_files (list): Sorted list of LiDAR binary files.
        label_files (list): Sorted list of label text files.
        fig (matplotlib.figure.Figure): Figure for animation.
        ax_img (matplotlib.axes.Axes): Axis for image display.
        ax_text (matplotlib.axes.Axes): Axis for summary text.
        img_display (matplotlib.image.AxesImage): Image display handle.
        scatter (matplotlib.collections.PathCollection): Scatter plot for LiDAR points.
        text_handle (matplotlib.text.Text): Text handle for summary.
        colors (dict): Mapping of track_id to RGB color.
        trajectories (dict): Track ID to list of 2D trajectory points.
        paused (bool): Pause state of animation.
        btn_pause (matplotlib.widgets.Button): Button to pause/play.
        anim (matplotlib.animation.FuncAnimation): Animation instance.
    """

    def __init__(self, num_frames=NUM_FRAMES):
        """
        Initialize the FusionAnimator with configuration, data paths, and matplotlib setup.

        Args:
            num_frames (int): Number of frames to animate.
        """
        self.num_frames = num_frames
        self.Tr_velo_to_cam, self.R_rect, self.P2 = load_calibration()
        self.tracker = MultiObjectTracker()

        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])[:self.num_frames]
        self.lidar_files = sorted([f for f in os.listdir(lidar_folder) if f.endswith('.bin')])[:self.num_frames]
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])[:self.num_frames]

        # Create figure with side-by-side subplots (image + summary text)
        self.fig, (self.ax_img, self.ax_text) = plt.subplots(1, 2, figsize=(15, 7),
                                                             gridspec_kw={'width_ratios': [3, 1]})

        # Initialize left axis for image + LiDAR scatter
        self.img_display = self.ax_img.imshow(np.zeros((375, 1242, 3), dtype=np.uint8))
        self.scatter = self.ax_img.scatter([], [], s=POINT_SIZE, c=[], cmap='jet', alpha=0.8)
        self.ax_img.axis('off')
        self.ax_img.set_title('KITTI Sensor Fusion with 3D Bounding Boxes & Object Tracking')

        # Initialize right axis for summary text
        self.ax_text.axis('off')
        self.text_handle = self.ax_text.text(
            0.5, 0.5, "",
            ha='center', va='center',
            fontsize=10,
            wrap=True,
            fontfamily='monospace',
            linespacing=1.2,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )

        # Pause/play button below right panel
        btn_ax = plt.axes([0.81, 0.01, 0.1, 0.05])
        self.btn_pause = Button(btn_ax, 'Pause')
        self.btn_pause.on_clicked(self.toggle_pause)
        self.paused = False

        self.colors = {}
        self.trajectories = {}

        # Create animation object
        self.anim = FuncAnimation(self.fig, self.update, frames=self.num_frames, interval=100, blit=False)

    def toggle_pause(self, event):
        """
        Toggle the paused state of the animation and update button label.

        Args:
            event: Mouse click event (ignored).
        """
        self.paused = not self.paused
        self.btn_pause.label.set_text('Play' if self.paused else 'Pause')

    def get_color(self, track_id):
        """
        Assign or retrieve consistent RGB color for each track ID.

        Args:
            track_id (int): Track ID number.

        Returns:
            tuple: RGB color tuple as integers (0-255).
        """
        if track_id not in self.colors:
            cmap = colormaps['tab20']
            color = cmap(track_id % 20)[:3]  # RGB floats 0-1
            self.colors[track_id] = tuple(int(255 * c) for c in color)
        return self.colors[track_id]

    def update(self, frame_idx):
        """
        Update the animation frame: load data, update tracking, draw bounding boxes and summary.

        Args:
            frame_idx (int): Index of current animation frame.

        Returns:
            tuple: Updated matplotlib artists.
        """
        if self.paused:
            return self.img_display, self.scatter, self.text_handle

        # Load camera image and convert to RGB
        img = cv2.imread(os.path.join(image_folder, self.image_files[frame_idx]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load LiDAR scan and project to image plane
        scan = np.fromfile(os.path.join(lidar_folder, self.lidar_files[frame_idx]), dtype=np.float32).reshape(-1, 4)
        pts_2d, colors = project_lidar_to_image(scan, self.Tr_velo_to_cam, self.R_rect, self.P2, color_by='intensity')

        # Load and parse label file for detected objects
        label_path = os.path.join(label_folder, self.label_files[frame_idx])
        objects = parse_label_file(label_path)

        # Extract 3D centroids from objects for tracking
        centroids = [obj['location'] for obj in objects]
        tracked = self.tracker.update(centroids)

        # Update trajectories and assign colors
        for track in tracked:
            track_id = track.track_id
            state = track.get_state()
            if state is None:
                continue
            pos_3d = state[:3]
            pts_proj = project_to_image(np.array([pos_3d]), self.P2)
            pos_2d = (int(pts_proj[0, 0]), int(pts_proj[0, 1]))
            self.trajectories.setdefault(track_id, []).append(pos_2d)
            self.get_color(track_id)

        # Update displayed image and LiDAR scatter points
        self.img_display.set_data(img_rgb)
        self.scatter.set_offsets(pts_2d)
        self.scatter.set_array(colors)

        # Draw 3D bounding boxes and IDs on image using OpenCV
        for obj, track in zip(objects, tracked):
            corners_3d = compute_3d_box(obj['dimensions'], obj['location'], obj['rotation_y'])
            pts_box = project_to_image(corners_3d, self.P2).astype(int)
            for i, j in zip([0,1,2,3], [1,2,3,0]):
                cv2.line(img_rgb, tuple(pts_box[i]), tuple(pts_box[j]), self.get_color(track.track_id), 2)
            for i, j in zip([4,5,6,7], [5,6,7,4]):
                cv2.line(img_rgb, tuple(pts_box[i]), tuple(pts_box[j]), self.get_color(track.track_id), 2)
            for i, j in zip(range(4), range(4,8)):
                cv2.line(img_rgb, tuple(pts_box[i]), tuple(pts_box[j]), self.get_color(track.track_id), 2)

            x_min, y_min = np.min(pts_box, axis=0)
            cv2.putText(img_rgb, f'ID {track.track_id}', (x_min, max(y_min - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.get_color(track.track_id), 2, cv2.LINE_AA)

        # Set updated image to display
        self.img_display.set_data(img_rgb)

        # Generate and display GPT scene summary text
        summary = generate_scene_summary(objects, tracked)
        wrapped = "\n".join(textwrap.wrap(summary, width=40))
        self.text_handle.set_text(wrapped)

        return self.img_display, self.scatter, self.text_handle

    def save_trajectory_image(self, filename='trajectory_summary.png'):
        """
        Save a static image visualizing object trajectories over the last frame.

        Args:
            filename (str): Filename to save the image.
        """
        img = cv2.imread(os.path.join(image_folder, self.image_files[-1]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for obj_id, points in self.trajectories.items():
            pts = np.array(points, dtype=np.int32)
            color = self.get_color(obj_id)
            for i in range(1, len(pts)):
                cv2.line(img_rgb, tuple(pts[i-1]), tuple(pts[i]), color, 3)
            final = pts[-1]
            cv2.putText(img_rgb, f'ID {obj_id}', (final[0], max(final[1] - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3, cv2.LINE_AA)

        plt.imsave(filename, img_rgb)
        print(f"Trajectory summary image saved as {filename}")

    def save_animation(self, filename='fusion_tracking.gif'):
        """
        Save the entire animation as a GIF file.

        Args:
            filename (str): Filename for the saved GIF.
        """
        print(f"Saving animation to {filename} ...")
        self.anim.save(filename, writer='pillow', fps=10)
        print("Animation saved.")

def main():
    """
    Main entry point to run the FusionAnimator visualization.
    """
    animator = FusionAnimator()
    plt.show()
    animator.save_animation()
    animator.save_trajectory_image()

if __name__ == "__main__":
    main()
