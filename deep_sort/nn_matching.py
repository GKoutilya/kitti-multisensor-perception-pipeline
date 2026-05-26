import numpy as np

class NearestNeighborGallery:
    def __init__(self, max_gallery_size=100):
        self.max_gallery_size = max_gallery_size
        self.gallery = {}

    def update(self, track_id, embedding):
        if track_id not in self.gallery:
            self.gallery[track_id] = []
        self.gallery[track_id].append(embedding)
        if len(self.gallery[track_id]) > self.max_gallery_size:
            self.gallery[track_id].pop(0)

    def delete(self, track_id):
        self.gallery.pop(track_id, None)

    