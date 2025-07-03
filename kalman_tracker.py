import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanTracker:
    def __init__(self, init_position, obj_id):
        self.track_id = obj_id  # <-- 🛠️ This is critical
        self.kf = self._init_kalman_filter(init_position)
        self.age = 1
        self.time_since_update = 0
        self.history = []

    def _init_kalman_filter(self, init_position):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.array([[1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        kf.R *= 0.1
        kf.P *= 10.
        kf.Q *= 0.01
        kf.x[:3] = np.array(init_position).reshape(3, 1)
        return kf

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:3].flatten()

    def update(self, measurement):
        self.kf.update(np.array(measurement).reshape(3, 1))
        self.time_since_update = 0

    def get_state(self):
        return self.kf.x[:3].flatten()

    def get_trajectory(self):
        """
        Get the history of tracked positions.

        Returns:
            list of np.ndarray: Past [x, y, z] positions.
        """
        return self.history


class MultiObjectTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        updated_tracks = []

        for det in detections:
            det = np.array(det)

            # Match with existing tracks (simple nearest neighbor)
            best_track = None
            best_dist = float('inf')
            for track in self.tracks:
                pred = track.get_state()
                if pred is None:
                    continue
                dist = np.linalg.norm(pred - det)
                if dist < best_dist and dist < 3.0:  # distance threshold
                    best_dist = dist
                    best_track = track

            if best_track:
                best_track.update(det)
                updated_tracks.append(best_track)
                self.tracks.remove(best_track)
            else:
                # Create new track
                new_track = KalmanTracker(det, self.next_id)
                updated_tracks.append(new_track)
                self.next_id += 1

        # Keep unmatched tracks (optional: you can implement track deletion for lost tracks here)
        self.tracks = updated_tracks

        # Return only tracks with valid track_id attribute
        return [t for t in self.tracks if hasattr(t, "track_id")]
