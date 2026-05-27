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

    def cost_matrix(self, embeddings, track_ids):
        n = embeddings.shape[0]
        m = len(track_ids)
        cost = np.full((n,m), 1.0, dtype=np.float32)
        for j, tid in enumerate(track_ids):
            if tid not in self.gallery or not self.gallery[tid]:
                continue
            G = np.array(self.gallery[tid])
            dots = embeddings @ G.T
            cost[:, j] = 1.0 - np.max(dots, axis=1)
        return cost