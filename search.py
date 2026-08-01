"""
Search module for performing image search queries.
"""
import os
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

from config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    DEFAULT_TOP_K, MAX_TOP_K, VALID_GENDERS,
    BASE_DIR
)
from dataset import get_transforms
from model import EmbeddingModel
from indexer import FAISSIndex


def load_image_from_url(url, timeout=10):
    """
    Load an image from a URL.
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        PIL Image or None if failed
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None


def load_image_from_path(path):
    """
    Load an image from a local file path.
    
    Args:
        path: Local file path
        
    Returns:
        PIL Image or None if failed
    """
    try:
        img = Image.open(path).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image from path: {e}")
        return None


def is_url(input_str):
    """Check if input string is a URL."""
    return input_str.startswith('http://') or input_str.startswith('https://')


def preprocess_image(image, transform=None):
    """
    Preprocess a PIL image for the model.
    
    Args:
        image: PIL Image
        transform: Optional transform pipeline
        
    Returns:
        Tensor ready for model input
    """
    if transform is None:
        transform = get_transforms()
    return transform(image).unsqueeze(0)


def search_image(query_input, faiss_index, model, k=DEFAULT_TOP_K, filter_gender=None, transform=None):
    """
    Search for similar images.
    
    Args:
        query_input: URL or local path to query image
        faiss_index: FAISSIndex instance
        model: EmbeddingModel instance
        k: Number of results to return
        filter_gender: Optional gender filter
        transform: Optional transform pipeline
        
    Returns:
        List of matching image IDs
    """
    # Load query image
    if is_url(query_input):
        image = load_image_from_url(query_input)
    else:
        image = load_image_from_path(query_input)
    
    if image is None:
        raise ValueError(f"Could not load image from: {query_input}")
    
    # Preprocess
    if transform is None:
        transform = get_transforms()
    input_tensor = preprocess_image(image, transform)
    
    # Get embedding
    query_vector = model.get_single_embedding(input_tensor)
    
    # Search
    results = faiss_index.search(query_vector, k=k, filter_gender=filter_gender)
    
    return results


def display_results(results, metadata_dict, dataset_path, k=10, cols=5):
    """
    Display search results as a grid of images.
    
    Args:
        results: List of image IDs
        metadata_dict: Metadata dictionary
        dataset_path: Path to dataset images directory
        k: Number of results to display
        cols: Number of columns in grid
    """
    display_count = min(len(results), k)
    if display_count == 0:
        print("No results to display")
        return
    
    rows = (display_count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle single row/col case
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    images_dir = os.path.join(dataset_path, 'images')
    
    for i, ax in enumerate(axes.flatten()):
        if i < display_count:
            img_id = results[i]
            formatted_id = str(img_id).rjust(5, '0')
            img_path = os.path.join(images_dir, f"{formatted_id}.jpg")
            
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                title = metadata_dict.get(img_id, {}).get('productDisplayName', '')[:30]
                gender = metadata_dict.get(img_id, {}).get('gender', 'Unknown')
                ax.set_title(f"{title}...\n({gender})", fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{img_id}", ha='center', va='center')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(display_count, rows * cols):
        axes.flatten()[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def search_and_display(query_input, faiss_index, model, metadata_dict, dataset_path, 
                       k=DEFAULT_TOP_K, filter_gender=None):
    """
    Complete search and display pipeline.
    
    Args:
        query_input: URL or local path to query image
        faiss_index: FAISSIndex instance
        model: EmbeddingModel instance
        metadata_dict: Metadata dictionary
        dataset_path: Path to dataset
        k: Number of results
        filter_gender: Optional gender filter
        
    Returns:
        List of matching image IDs
    """
    # Validate gender filter
    if filter_gender and filter_gender not in VALID_GENDERS:
        print(f"Warning: '{filter_gender}' is not a valid gender. Valid options: {VALID_GENDERS}")
        filter_gender = None
    
    print(f"Searching for: {query_input}")
    if filter_gender:
        print(f"Filtering by gender: {filter_gender}")
    
    results = search_image(query_input, faiss_index, model, k=k, filter_gender=filter_gender)
    
    print(f"Found {len(results)} results")
    display_results(results, metadata_dict, dataset_path, k=k)
    
    return results