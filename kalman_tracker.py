import numpy as np
from filterpy.kalman import KalmanFilter

"""
    Used when you need to track objects that move over time, not just detecting them once.
    Kalman filters make smart guesses that keep estimating where things are.
"""


"""
    We create a tracker for each object we find in 3D space.
    The tracker keeps track of:
        Where the object is now,
        How fast it's moving, and
        Where it'll be next
    Each tracker gets a unique ID
"""
class KalmanTracker:
    # Stores track_id and initializes a Kalman filter and metadata (age, update time, history)
    def __init__(self, init_position, obj_id):
        self.track_id = obj_id  # Necessary
        self.kf = self._init_kalman_filter(init_position)
        self.age = 1
        self.time_since_update = 0
        self.history = []
    
    # State vector: 6D [x, y, z, vx, vy, vz], Measurement vector: 3D [x, y, z]
    def _init_kalman_filter(self, init_position):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        # Constant velocity motion model
        kf.F = np.array([[1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        # Observation only sees position
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        kf.R *= 0.1 # Measurement noise - tuned low (0.1) for confident measures
        kf.P *= 10. # Initial Uncertainty - high (10.), good choice.
        kf.Q *= 0.01 # Process noise - tuned low (0.01), reasonable
        kf.x[:3] = np.array(init_position).reshape(3, 1)
        return kf
    
    def predict(self):
        """
            Advances the state using the motion model.
            If the car is here and moving that fast, it'll probably be here next.
            Tracker already has a guess before we get the next camera/LIDAR reading.
        """
        self.kf.predict()
        # Increments age and time_since_update
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:3].flatten() # Returns the predicted [x, y, z]
    
    def update(self, measurement):
        """
            Takes in a 3D detection and corrects the prediction and resets the time_since_update
            When the sensors push new information, the tracker updates its guess.
                "Actually, the object was detected here," "Oh! Okay, let me fix my position."
            The cycle of predicting and correcting its prediction makes the tracking smooth and robust.
        """
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
    """
        Team manager/boss of all the little trackers:
            Gets all new 3D positions of detected objects
            Matches each one to an existing tracker (using closest distance)
            Updates the tracker with the new position
            Or creates a new tracker if it can't match
        This is how it handles multiple moving objects all at once, like cars and people.
    """
    # Stores active tracks and a counter for assigning new IDs
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        """
           We only keep the trackers that are active and useful.
           Only return trackers that:
                Are still tracking something
                Have a valid ID (were actually matched or initialized) 
        """
        updated_tracks = []
        # For each new detection:
        for det in detections:
            det = np.array(det)

            # Match with existing tracks (simple nearest neighbor)
            # Tries to find the nearest track (Euclidean distance)
            # If there's a match within 3.0 meters, it updates the track
            # If there's no match, it initializes a new track
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

        # Only matched or new tracks are kept
        self.tracks = updated_tracks

        # Return only tracks with valid track_id attribute
        return [t for t in self.tracks if hasattr(t, "track_id")]