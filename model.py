"""
Model module for loading and managing the ResNet50 embedding model.
"""
import torch
import torchvision.models as models
from config import MODEL_NAME, EMBEDDING_DIM, DEVICE, MODEL_DIR


class EmbeddingModel:
    """Wrapper for the ResNet50 embedding model."""
    
    def __init__(self, device=None):
        self.device = device or DEVICE
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the pretrained ResNet50 model and modify for embeddings."""
        print(f"Loading {MODEL_NAME} model on {self.device}...")
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # Replace final FC layer with Identity to get 2048-dim embeddings
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        print(f"Model loaded successfully. Embedding dimension: {EMBEDDING_DIM}")
    
    def get_embeddings(self, images):
        """
        Get normalized embeddings for a batch of images.
        
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Normalized embeddings of shape (batch_size, EMBEDDING_DIM)
        """
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            # Normalize for cosine similarity (FAISS IndexFlatIP)
            normalized = torch.nn.functional.normalize(outputs, p=2, dim=1)
        return normalized.cpu()
    
    def get_single_embedding(self, image_tensor):
        """
        Get embedding for a single image tensor.
        
        Args:
            image_tensor: Tensor of shape (3, H, W) or (1, 3, H, W)
            
        Returns:
            Normalized embedding of shape (EMBEDDING_DIM,)
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        embedding = self.get_embeddings(image_tensor)
        return embedding.squeeze(0).numpy().astype('float32')


def load_model(device=None):
    """Factory function to create and load the embedding model."""
    return EmbeddingModel(device=device)