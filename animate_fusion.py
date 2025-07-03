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

NUM_FRAMES = 25  # You can increase this, but memory and CPU usage grow linearly
LIDAR_DOWNSAMPLE = 5
POINT_SIZE = 1.5

def read_calib_file(filepath):
    """
    Parse KITTI calibration file into dictionary of numpy arrays.

    Args:
        filepath (str): Path to calibration file.

    Returns:
        dict: Dictionary with keys as calibration parameters and values as numpy arrays.
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
        tuple: (Tr_velo_to_cam, R_rect, P2) transformation matrices as numpy arrays.
            - Tr_velo_to_cam (4x4): LIDAR to camera coordinate transform.
            - R_rect (4x4): Rectification matrix.
            - P2 (3x4): Camera projection matrix.
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
        scan (np.array): Nx4 LIDAR points [x, y, z, intensity].
        Tr_velo_to_cam (np.array): 4x4 LIDAR to camera transform.
        R_rect (np.array): 4x4 rectification matrix.
        P2 (np.array): 3x4 projection matrix.
        color_by (str): 'depth' or 'intensity' determines point coloring.

    Returns:
        tuple:
            - pts_2d (np.array): Nx2 array of 2D pixel coordinates.
            - colors (np.array): N-length array of colors corresponding to points.
    """
    scan = scan[::LIDAR_DOWNSAMPLE]
    scan = scan[scan[:, 0] > 0]  # Filter points in front of vehicle

    lidar_hom = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1))))
    pts_cam = (Tr_velo_to_cam @ lidar_hom.T).T
    pts_rect = (R_rect @ pts_cam.T).T
    pts_rect = pts_rect[pts_rect[:, 2] > 0]

    pts_2d = (P2 @ pts_rect.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2, np.newaxis]

    if color_by == 'intensity':
        colors = scan[:, 3]
    else:
        colors = pts_rect[:, 2]  # depth

    return pts_2d, colors

class FusionAnimator:
    """
    Class to animate LIDAR and camera sensor fusion from KITTI dataset.

    Attributes:
        num_frames (int): Number of frames to animate.
        Tr_velo_to_cam (np.array): LIDAR to camera transform.
        R_rect (np.array): Rectification matrix.
        P2 (np.array): Camera projection matrix.
        image_files (list): Sorted list of image filenames.
        lidar_files (list): Sorted list of LIDAR filenames.
        fig, ax (matplotlib objects): Figure and axis for plotting.
        img_display (matplotlib image): Image display object.
        scatter (matplotlib scatter): Scatter plot for LIDAR points.
        paused (bool): Animation pause state.
        anim (FuncAnimation): Matplotlib animation object.
    """
    def __init__(self, num_frames=NUM_FRAMES):
        """
        Initialize the animator, load calibration, and prepare plot.

        Args:
            num_frames (int): Number of frames to animate.
        """
        self.num_frames = num_frames
        self.Tr_velo_to_cam, self.R_rect, self.P2 = load_calibration()
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])[:num_frames]
        self.lidar_files = sorted([f for f in os.listdir(lidar_folder) if f.endswith('.bin')])[:num_frames]

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.img_display = self.ax.imshow(np.zeros((375, 1242, 3), dtype=np.uint8))
        self.scatter = self.ax.scatter([], [], s=POINT_SIZE, c=[], cmap='jet', alpha=0.8)
        self.ax.axis('off')
        self.ax.set_title('KITTI Sensor Fusion: LIDAR projected onto Camera')

        # Play/Pause button
        axpause = plt.axes([0.81, 0.01, 0.1, 0.05])
        self.btn_pause = Button(axpause, 'Pause')
        self.btn_pause.on_clicked(self.toggle_pause)
        self.paused = False

        self.anim = FuncAnimation(self.fig, self.update, frames=self.num_frames,
                                  interval=100, blit=False)
    
    def toggle_pause(self, event):
        """
        Toggle the pause/play state of the animation.

        Args:
            event: Button click event (unused).
        """
        self.paused = not self.paused
        self.btn_pause.label.set_text('Play' if self.paused else 'Pause')

    def update(self, frame_idx):
        """
        Update the plot for the current frame index.

        Args:
            frame_idx (int): Frame index.

        Returns:
            tuple: Updated image and scatter plot objects.
        """
        if self.paused:
            return self.img_display, self.scatter

        img = cv2.imread(os.path.join(image_folder, self.image_files[frame_idx]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        scan = np.fromfile(os.path.join(lidar_folder, self.lidar_files[frame_idx]), dtype=np.float32).reshape(-1, 4)
        pts_2d, colors = project_lidar_to_image(scan, self.Tr_velo_to_cam, self.R_rect, self.P2, color_by='intensity')

        self.img_display.set_data(img_rgb)
        self.scatter.set_offsets(pts_2d)
        self.scatter.set_array(colors)

        # Add coordinate axis lines (static example)
        self.ax.plot([50, 100], [50, 50], color='r', linewidth=2, label='X axis')
        self.ax.plot([50, 50], [50, 100], color='g', linewidth=2, label='Y axis')

        return self.img_display, self.scatter

    def save_animation(self, filename='fusion_animation.mp4'):
        """
        Save the animation as an MP4 video file.

        Args:
            filename (str): Output filename.
        """
        print(f"Saving animation to {filename} ...")
        self.anim.save(filename, writer='ffmpeg', fps=10)
        print("Saved.")

def main():
    """
    Main function to run the animation.
    """
    animator = FusionAnimator()
    plt.show()
    # Uncomment the following line to save the animation video (requires ffmpeg)
    # animator.save_animation()

if __name__ == "__main__":
    main()
