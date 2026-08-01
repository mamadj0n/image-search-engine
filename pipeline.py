"""
Main pipeline module for the Image Search system.
Orchestrates the full workflow: dataset preparation, embedding generation, index building, and search.
"""
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath("."))

from config import DEVICE, INDEX_DIR
from dataset import prepare_dataset, load_metadata
from model import EmbeddingModel
from indexer import build_index, load_index, FAISSIndex
from search import search_and_display


class ImageSearchPipeline:
    """Main pipeline for the image search system."""
    
    def __init__(self, device=None):
        self.device = device or DEVICE
        self.dataset_path = None
        self.metadata = None
        self.valid_ids = None
        self.metadata_dict = None
        self.model = None
        self.faiss_index = None
        self.dataloader = None
    
    def setup_dataset(self):
        """Download dataset, load metadata, validate images."""
        print("=" * 60)
        print("STEP 1: Setting up dataset")
        print("=" * 60)
        self.dataset_path, self.metadata, self.valid_ids, self.dataloader = prepare_dataset()
        self.metadata_dict = self.metadata.set_index('id').to_dict('index')
        print(f"Dataset ready: {len(self.valid_ids)} valid images")
        return self
    
    def setup_model(self):
        """Load the embedding model."""
        print("=" * 60)
        print("STEP 2: Loading embedding model")
        print("=" * 60)
        self.model = EmbeddingModel(device=self.device)
        return self
    
    def build_index(self, force_rebuild=False):
        """Build or load the FAISS index."""
        print("=" * 60)
        print("STEP 3: Building/loading FAISS index")
        print("=" * 60)
        
        index_path = INDEX_DIR / "fashion_faiss.index"
        ids_path = INDEX_DIR / "image_ids.npy"
        
        if not force_rebuild and index_path.exists() and ids_path.exists():
            print("Loading existing index...")
            self.faiss_index = load_index()
            self.faiss_index.set_metadata(self.metadata_dict)
        else:
            print("Building new index...")
            self.faiss_index = self._build_index_from_scratch()
        
        print(f"Index ready with {self.faiss_index.index.ntotal} vectors")
        return self
    
    def _build_index_from_scratch(self):
        """Build FAISS index by extracting embeddings from all images."""
        if self.model is None:
            self.setup_model()
        if self.dataloader is None:
            self.setup_dataset()
        
        all_embeddings = []
        all_image_ids = []
        
        print("Extracting embeddings...")
        for images, ids in tqdm(self.dataloader, desc="Processing batches"):
            embeddings = self.model.get_embeddings(images)
            all_embeddings.append(embeddings.numpy())
            all_image_ids.extend(ids.tolist())
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        all_image_ids = np.array(all_image_ids)
        
        print(f"Extracted embeddings shape: {all_embeddings.shape}")
        
        # Build and save index
        faiss_index = build_index(
            all_embeddings, 
            all_image_ids, 
            self.metadata_dict
        )
        
        return faiss_index
    
    def search(self, query_input, k=10, filter_gender=None):
        """
        Search for similar images.
        
        Args:
            query_input: URL or local path to query image
            k: Number of results
            filter_gender: Optional gender filter
            
        Returns:
            List of matching image IDs
        """
        if self.faiss_index is None:
            self.build_index()
        if self.model is None:
            self.setup_model()
        
        return search_and_display(
            query_input=query_input,
            faiss_index=self.faiss_index,
            model=self.model,
            metadata_dict=self.metadata_dict,
            dataset_path=self.dataset_path,
            k=k,
            filter_gender=filter_gender
        )
    
    def run_full_pipeline(self, query_input=None, k=10, filter_gender=None, force_rebuild=False):
        """Run the complete pipeline from scratch."""
        self.setup_dataset()
        self.setup_model()
        self.build_index(force_rebuild=force_rebuild)
        
        if query_input:
            return self.search(query_input, k=k, filter_gender=filter_gender)
        return None


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Search Pipeline")
    parser.add_argument("--query", type=str, help="Query image URL or path")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--gender", type=str, help="Filter by gender (Men, Women, Unisex, Boys, Girls)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    
    # تغییر این خط: استفاده از parse_known_args به جای parse_args
    args, _ = parser.parse_known_args()
    
    device = None if args.device == "auto" else args.device
    
    pipeline = ImageSearchPipeline(device=device)
    pipeline.run_full_pipeline(
        query_input=args.query,
        k=args.k,
        filter_gender=args.gender,
        force_rebuild=args.rebuild
    )


if __name__ == "__main__":
    main()