import numpy as np
from scipy.optimize import linear_sum_assignment # Hungarian Algorithm Implementation
from filterpy.kalman import KalmanFilter
from deep_sort.nn_matching import NearestNeighborGallery
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

CHI2_THRESHOLD = 7.815
MAX_AGE = 30
MIN_HITS = 2
MAX_COSINE_DISTANCE = 0.7

class KalmanTracker:
    # Stores track_id and initializes a Kalman filter and metadata (age, update time, history)
    def __init__(self, init_position, obj_id):
        self.track_id = obj_id  # Necessary
        self.kf = self._init_kalman_filter(init_position)
        self.age = 1
        self.time_since_update = 0
        self.history = [np.array(init_position).flatten()]
        self.hits = 1
        self.state = 'tentative'
    
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
        self.history.append(self.kf.x[:3].flatten())
        self.hits += 1
        if self.state == 'tentative' and self.hits >= MIN_HITS:
            self.state = 'confirmed'

    def mark_missed(self):
        if self.state == 'tentative':
            self.state = 'deleted'
        elif self.time_since_update > MAX_AGE:
            self.state = 'deleted'

    def mahalanobis_distance(self, detection):
        z = np.array(detection).reshape(3, 1)
        H = self.kf.H
        P = self.kf.P
        R = self.kf.R
        S = H @ P @ H.T + R
        innov = z - self.kf.x[:3]
        return float(innov.T @ np.linalg.inv(S) @ innov)

    def get_state(self): return self.kf.x[:3].flatten()

    def get_velocity(self): return self.kf.x[3:6].flatten()

    def get_trajectory(self): return self.history

    def is_tentative(self): return self.state == 'tentative'

    def is_confirmed(self): return self.state == 'confirmed'

    def is_deleted(self): return self.state == 'deleted'


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
    def __init__(self, max_cosine_distance=MAX_COSINE_DISTANCE, nn_budget=100):
        self.tracks = []
        self.next_id = 0
        self.gallery = NearestNeighborGallery(max_gallery_size=nn_budget)
        self.max_cosine_distance = max_cosine_distance

    def update(self, detections, embeddings):
        # 1. Predict all tracks forward
        for track in self.tracks:
            track.predict()

        confirmed = [t for t in self.tracks if t.is_confirmed()]
        tentative = [t for t in self.tracks if t.is_tentative()]

        # 2. Cascade match detections against confirmed tracks
        matches, unmatched_dets, _ = self._cascade_match(detections, embeddings, confirmed)

        # 3. Distance match leftover detections against tentative tracks
        leftover_positions = [detections[i] for i in unmatched_dets]
        tent_matches, unmatched_dets, _ = self._distance_match(leftover_positions, tentative, unmatched_dets)
        matches += tent_matches

        # 4. Update matched tracks
        for det_idx, track in matches:
            track.update(detections[det_idx])
            self.gallery.update(track.track_id, embeddings[det_idx])

        # 5. Mark unmatched tracks as missed
        matched_tracks = {track for _, track in matches}
        for track in self.tracks:
            if track not in matched_tracks:
                track.mark_missed()

        # 6. Spawn new tracks for unmatched detections, delete dead tracks
        for det_idx in unmatched_dets:
            new_track = KalmanTracker(detections[det_idx], self.next_id)
            self.gallery.update(self.next_id, embeddings[det_idx])
            self.tracks.append(new_track)
            self.next_id += 1

        for track in [t for t in self.tracks if t.is_deleted()]:
            self.gallery.delete(track.track_id)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return [t for t in self.tracks if t.is_confirmed()]
    
    def _cascade_match(self, detections, embeddings, confirmed_tracks):
        if not confirmed_tracks or not detections:
            return [], list(range(len(detections))), confirmed_tracks
        
        track_ids = [t.track_id for t in confirmed_tracks]

        # Appearance cost matrix from the gallery
        app_cost = self.gallery.cost_matrix(embeddings, track_ids)

        # Mahalanobis gating - mask implausible pairs
        for j, track in enumerate(confirmed_tracks):
            for i, det in enumerate(detections):
                if track.mahalanobis_distance(det) > CHI2_THRESHOLD:
                    app_cost[i, j] = 1.0
        
        # Appearance gate - mask dissimilar pairs
        app_cost[app_cost > self.max_cosine_distance] = 1.0

        # Hungarain Assignment
        row_inds, col_inds = linear_sum_assignment(app_cost)

        matches, unmatched_dets, unmatched_tracks = [], [], []
        matched_cols = set()

        for r, c in zip(row_inds, col_inds):
            if app_cost[r, c] >= 1.0: # gated pair, treat as unmatched
                unmatched_dets.append(r)
            else:
                matches.append((r, confirmed_tracks[c]))
                matched_cols.add(c)

        matched_rows = {r for r, _ in matches}
        for i in range(len(detections)):
            if i not in matched_rows and i not in unmatched_dets:
                unmatched_dets.append(i)
        for j, track in enumerate(confirmed_tracks):
            if j not in matched_cols:
                unmatched_tracks.append(track)

        return matches, unmatched_dets, unmatched_tracks
    
    def _distance_match(self, det_positions, tentative_tracks, original_indices):
        if not tentative_tracks or not det_positions:
            return [], original_indices, tentative_tracks
        
        matches, unmatched = [], list(original_indices)
        for track in tentative_tracks:
            pred = track.get_state()
            best_idx, best_dist = None, float('inf')
            for orig_idx, det in zip(original_indices, det_positions):
                if orig_idx not in unmatched:
                    continue
                dist = np.linalg.norm(pred - np.array(det))
                if dist < best_dist and dist < 3.0:
                    best_dist, best_idx = dist, orig_idx
            if best_idx is not None:
                matches.append((best_idx, track))
                unmatched.remove(best_idx)

        return matches, unmatched, tentative_tracks