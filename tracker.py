import numpy as np
from scipy.spatial import distance

class SimpleTracker:
    """
    A simple object tracker that assigns persistent IDs based on centroid distance.
    """
    def __init__(self, max_distance=2.0):
        """
        Initialize the tracker.

        Args:
            max_distance (float): Maximum allowed distance (in meters) to consider two detections the same object.
        """
        self.next_id = 0
        self.tracks = {}  # id -> {'centroid': np.array([x, y, z]), 'history': [centroids]}
        self.max_distance = max_distance

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections (list): List of detections, each detection is a tuple or list: [x, y, z]

        Returns:
            list: List of dicts with keys: id, position (3D), and trajectory history
        """
        detections = [np.array(det) for det in detections]
        assignments = {}  # detection index -> track_id

        # Track assignment via nearest neighbor
        unmatched_detections = set(range(len(detections)))
        for track_id, track_data in self.tracks.items():
            min_dist = float('inf')
            best_idx = None
            for idx in unmatched_detections:
                dist = np.linalg.norm(detections[idx] - track_data['centroid'])
                if dist < min_dist and dist <= self.max_distance:
                    min_dist = dist
                    best_idx = idx

            if best_idx is not None:
                # Assign detection to this track
                self.tracks[track_id]['centroid'] = detections[best_idx]
                self.tracks[track_id]['history'].append(detections[best_idx])
                assignments[best_idx] = track_id
                unmatched_detections.remove(best_idx)

        # Create new tracks for unmatched detections
        for idx in unmatched_detections:
            self.tracks[self.next_id] = {
                'centroid': detections[idx],
                'history': [detections[idx]]
            }
            assignments[idx] = self.next_id
            self.next_id += 1

        # Prepare tracked output
        results = []
        for idx, det in enumerate(detections):
            obj_id = assignments[idx]
            history = self.tracks[obj_id]['history']
            results.append({
                'id': obj_id,
                'position': det,
                'trajectory': history
            })

        return results
