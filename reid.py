import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
import ssl
import warnings
import time

# Bypass potential urllib SSL / Proxy handshake issues on localized networks
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

class ReIDEngine:
    def __init__(self, use_cuda=True, backbone_type="mobilenet_v2"):
        """
        Lightweight Deep Re-Identification Engine supporting multiple backbones.
        Upgraded with FP16 Half-Precision, Batched Inference, and Dynamic Swapping.
        
        backbone_type: "mobilenet_v2" (Ultra-Fast) or "resnet18" (Accurate)
        """
        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        self.fp16 = (self.device.type == "cuda") # FP16 optimization on Nvidia GPUs
        self.backbone_type = backbone_type
        
        print(f"Re-ID Embedding Engine initialized on device: {self.device} (FP16: {self.fp16}) | Model: {self.backbone_type}")
        
        # Diagnostic tracking
        self.last_redetection_time = 0.0
        
        try:
            if self.backbone_type == "resnet18":
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                # Strip final classification layer (Residual Average Pool remains)
                self.model = nn.Sequential(*list(backbone.children())[:-1])
                print("Successfully loaded pre-trained ResNet-18 weights.")
            elif self.backbone_type == "mobilenet_v2":
                backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                # Strip final classifier layer and preserve spatial feature maps
                self.model = nn.Sequential(
                    backbone.features,
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                print("Successfully loaded pre-trained MobileNetV2 weights.")
            else:
                raise ValueError(f"Unsupported backbone type: {self.backbone_type}")
        except Exception as e:
            warnings.warn(f"Network error downloading pre-trained weights ({e}). Using random initialization.")
            if self.backbone_type == "resnet18":
                backbone = models.resnet18(weights=None)
                self.model = nn.Sequential(*list(backbone.children())[:-1])
            else:
                backbone = models.mobilenet_v2(weights=None)
                self.model = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d((1, 1)))
        
        self.model.to(self.device)
        self.model.eval()
        
        # Performance optimization: Convert weights to half-precision if on CUDA
        if self.fp16:
            self.model = self.model.half()
        
        # Re-ID aspect ratio cropping & standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.target_gallery = {}

    @torch.no_grad()
    def extract_embedding(self, crop):
        """
        Extracts a normalized feature embedding vector for a single bounding box crop.
        """
        if crop is None or crop.size == 0:
            return None
            
        crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        if self.fp16:
            tensor = tensor.half()
        
        embedding = self.model(tensor).flatten()
        embedding = embedding / torch.norm(embedding)
        
        return embedding.cpu().numpy()

    def register_target(self, target_id, frame, bbox):
        """
        Extracts and registers the reference target vector into the Re-ID search gallery.
        """
        x, y, w, h = map(int, bbox)
        height, width, _ = frame.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
            
        embedding = self.extract_embedding(crop)
        if embedding is not None:
            self.target_gallery[target_id] = embedding
            print(f"Target '{target_id}' successfully registered in the visual Re-ID gallery.")
            return True
        return False

    def compare_similarity(self, embedding_a, embedding_b):
        """
        Calculates cosine similarity between two feature arrays.
        """
        if embedding_a is None or embedding_b is None:
            return 0.0
        return float(np.dot(embedding_a, embedding_b))

    def find_best_match(self, frame, candidates_bbox, target_id, similarity_threshold=0.75, return_time=False):
        """
        Executes parallel GPU batched inference across all candidate regions.
        Guaranteed to ALWAYS return a valid tuple containing details to prevent GUI thread unpacking crashes.
        """
        start_time = time.perf_counter()
        
        if target_id not in self.target_gallery or not candidates_bbox:
            elapsed_time = (time.perf_counter() - start_time) * 1000.0
            self.last_redetection_time = elapsed_time
            # Safe tuple unpack fallback 1
            return (None, 0.0, elapsed_time) if return_time else (None, 0.0)
            
        target_embedding = self.target_gallery[target_id]
        height, width, _ = frame.shape
        
        valid_crops = []
        valid_bboxes = []
        
        # 1. Pre-process crops on CPU
        for bbox in candidates_bbox:
            x, y, w, h = map(int, bbox)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
                valid_crops.append(self.transform(crop_rgb))
                valid_bboxes.append(bbox)
                
        if not valid_crops:
            elapsed_time = (time.perf_counter() - start_time) * 1000.0
            self.last_redetection_time = elapsed_time
            # Safe tuple unpack fallback 2
            return (None, 0.0, elapsed_time) if return_time else (None, 0.0)
            
        # 2. Batch and stack to offload directly to GPU
        tensor_batch = torch.stack(valid_crops).to(self.device)
        if self.fp16:
            tensor_batch = tensor_batch.half()
            
        # 3. Fast Parallel feature extraction
        with torch.no_grad():
            embeddings = self.model(tensor_batch).view(tensor_batch.size(0), -1)
            embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
            embeddings_np = embeddings.cpu().numpy()
            
        # 4. Compare similarities
        best_bbox = None
        best_score = 0.0
        
        for i, emb in enumerate(embeddings_np):
            similarity = self.compare_similarity(target_embedding, emb)
            if similarity > best_score:
                best_score = similarity
                best_bbox = valid_bboxes[i]
                
        elapsed_time = (time.perf_counter() - start_time) * 1000.0
        self.last_redetection_time = elapsed_time
        
        if best_score >= similarity_threshold:
            print(f"Re-ID Match Confirmed! Score: {best_score:.3f} | Batch Time: {elapsed_time:.2f} ms")
            return (best_bbox, best_score, elapsed_time) if return_time else (best_bbox, best_score)
            
        print(f"Re-ID Sweep Failed. Best similarity: {best_score:.3f} | Batch Time: {elapsed_time:.2f} ms")
        # Safe tuple unpack fallback 3 (Prevents TypeError: cannot unpack non-iterable NoneType object)
        return (None, best_score, elapsed_time) if return_time else (None, best_score)