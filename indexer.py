"""
Index module for FAISS index creation, management, and searching.
"""
import faiss
import numpy as np
import torch
from pathlib import Path
from config import (
    EMBEDDING_DIM, INDEX_TYPE, INDEX_FILE, IDS_FILE,
    INDEX_DIR, VALID_GENDERS
)


class FAISSIndex:
    """Wrapper for FAISS index with metadata filtering."""
    
    def __init__(self, dimension=EMBEDDING_DIM, index_type=INDEX_TYPE):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.image_ids = None
        self.metadata_dict = None
    
    def create_index(self):
        """Create a new FAISS index."""
        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        print(f"Created {self.index_type} index with dimension {self.dimension}")
    
    def add_vectors(self, vectors, image_ids):
        """
        Add vectors to the index.
        
        Args:
            vectors: numpy array of shape (n, dimension), must be normalized for IndexFlatIP
            image_ids: list or array of image IDs corresponding to vectors
        """
        if self.index is None:
            self.create_index()
        
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        self.index.add(vectors)
        self.image_ids = np.array(image_ids)
        print(f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}")
    
    def save(self, index_dir=INDEX_DIR):
        """Save index and image IDs to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = index_dir / INDEX_FILE
        ids_path = index_dir / IDS_FILE
        
        faiss.write_index(self.index, str(index_path))
        np.save(ids_path, self.image_ids)
        print(f"Saved index to {index_path} and IDs to {ids_path}")
    
    def load(self, index_dir=INDEX_DIR):
        """Load index and image IDs from disk."""
        index_dir = Path(index_dir)
        index_path = index_dir / INDEX_FILE
        ids_path = index_dir / IDS_FILE
        
        self.index = faiss.read_index(str(index_path))
        self.image_ids = np.load(ids_path)
        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.image_ids)} IDs")
    
    def set_metadata(self, metadata_dict):
        """Set metadata dictionary for filtering."""
        self.metadata_dict = metadata_dict
    
    def search(self, query_vector, k=10, filter_gender=None):
        """
        Search the index for similar vectors.
        
        Args:
            query_vector: numpy array of shape (dimension,) or (1, dimension)
            k: number of results to return
            filter_gender: optional gender filter ('Men', 'Women', 'Unisex', etc.)
            
        Returns:
            List of image IDs of matching results
        """
        if self.index is None:
            raise ValueError("Index not loaded or created")
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = np.ascontiguousarray(query_vector.astype('float32'))
        
        # Search for more than k to account for filtering
        search_k = min(k * 3, self.index.ntotal) if filter_gender else k
        distances, indices = self.index.search(query_vector, search_k)
        
        filtered_results = []
        for idx in indices[0]:
            if idx >= len(self.image_ids) or idx < 0:
                continue
                
            img_id = self.image_ids[idx]
            
            # Apply gender filter if specified
            if filter_gender and self.metadata_dict:
                item_meta = self.metadata_dict.get(img_id, {})
                if item_meta.get('gender') != filter_gender:
                    continue
            
            filtered_results.append(img_id)
            
            if len(filtered_results) >= k:
                break
        
        return filtered_results


def build_index(embeddings, image_ids, metadata_dict, index_dir=INDEX_DIR):
    """
    Convenience function to build and save a FAISS index.
    
    Args:
        embeddings: numpy array of shape (n, dimension)
        image_ids: list of image IDs
        metadata_dict: dictionary mapping image_id to metadata
        index_dir: directory to save index
        
    Returns:
        FAISSIndex instance
    """
    faiss_index = FAISSIndex()
    faiss_index.create_index()
    faiss_index.add_vectors(embeddings, image_ids)
    faiss_index.set_metadata(metadata_dict)
    faiss_index.save(index_dir)
    return faiss_index


def load_index(index_dir=INDEX_DIR):
    """Load a FAISS index from disk."""
    faiss_index = FAISSIndex()
    faiss_index.load(index_dir)
    return faiss_index