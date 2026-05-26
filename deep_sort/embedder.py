import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models # Gives pretrained models, in this case: MobileNetV2
import torchvision.transforms as T # Handles image preprocessing pipeline

class AppearanceEmbedder:
    EMBED_DIM = 1280

    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu') # Auto-selects GPU if available, CPU otherwise
        self._build_model()
        self.transform = T.Compose([
            T.ToPILImage(), # Converts NumPy crop to PIL format for PyTorch
            T.Resize((128,64)), # Height x Width
            T.ToTensor(), # Converts to (3, H, W) float tensor in [0, 1]
            T.Normalize(mean=[0.485,0.456, 0.406], std=[0.229,0.224,0.225]) # Normalizes input for MobileNetV2
        ])

    def _build_model(self):
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.model = nn.Sequential(
            backbone.features, # Convolutional part of MobileNetV2 - everything up to but not including classifier
            nn.AdaptiveAvgPool2d((1, 1)), # Collapse spatial 4x2 grid into a single 1x1 value per channel - global average
            nn.Flatten(), # Squeezes is to 1 number per feature, 1 vector per crop
        ).to(self.device)
        self.model.eval() # Disables dropout and batch norm running-stat updates

    def _crop(self, img_rgb, bbox):
        h, w = img_rgb.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return img_rgb[y1:y2, x1:x2]
    
    @torch.no_grad() # Tells PyTorch not to build a computational graph
    def embed(self, img_rgb, bboxes):
        if not bboxes:
            return np.zeros((0, self.EMBED_DIM), dtype=np.float32)
        
        tensors, valid = [], []
        for bbox in bboxes:
            crop = self._crop(img_rgb, bbox)
            if crop is None:
                tensors.append(torch.zeros(3, 128, 64))
                valid.append(False)
            else:
                tensors.append(self.transform(crop))
                valid.append(True)

        batch = torch.stack(tensors).to(self.device)
        features = self.model(batch).cpu().numpy().astype(np.float32)

        for i, ok in enumerate(valid):
            if ok:
                norm = np.linalg.norm(features[i])
                if norm > 0:
                    features[i] /= norm
            else:
                features[i] = 0.0

        return features