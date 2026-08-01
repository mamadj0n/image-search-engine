# Image Search Engine

A powerful image search engine built with Streamlit, ResNet50, and FAISS for fast similarity search. Upload an image or provide an image URL to find visually similar products from the Fashion Product Images dataset.

## Features

- **Deep Learning Features**: Uses ResNet50 (pre-trained on ImageNet) to extract rich image features
- **Fast Similarity Search**: FAISS index for efficient similarity search (millions of vectors in milliseconds)
- **Gender Filtering**: Filter results by gender categories (Men, Women, Unisex, Boys, Girls)
- **Flexible Input**: Upload images or provide image URLs
- **Visual Results**: Beautiful grid display of search results with similarity scores
- **GPU Acceleration**: Automatic GPU detection and utilization when available
- **Index Persistence**: FAISS index saved to disk for fast subsequent searches
- **Modular Design**: Clean separation of concerns for easy maintenance and extension

## How It Works

1. **Feature Extraction**: Uses a pre-trained ResNet50 model (without the final classification layer) to extract 2048-dimensional feature vectors from images
2. **Similarity Search**: Stores these feature vectors in a FAISS index for efficient similarity search using inner product (cosine similarity after normalization)
3. **Search Process**: 
   - Extract features from query image
   - Search FAISS index for k-nearest neighbors
   - Return similar images with similarity scores and metadata

## Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-search-engine.git
cd image-search-engine

# Install dependencies
pip install -r requirements.txt

# Download dataset (first run only)
# The dataset will be automatically downloaded via kagglehub on first run
```

### Google Colab

1. Upload the entire `image-search-engine` folder to Colab
2. Install requirements: `!pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Usage

### As a Python Module

```python
from pipeline import ImageSearchPipeline

# Create pipeline
pipeline = ImageSearchPipeline()

# Run full pipeline with a query
results = pipeline.run_full_pipeline(
    query_input="https://example.com/image.jpg",  # URL or local path
    k=10,                                          # Number of results
    filter_gender="Men",                           # Optional: "Men", "Women", "Unisex", "Boys", "Girls"
    force_rebuild=False                            # Set True to rebuild index
)

# Or run steps individually
pipeline.setup_dataset()
pipeline.setup_model()
pipeline.build_index()
results = pipeline.search("path/to/image.jpg", k=5, filter_gender="Women")
```

### Command Line Interface

```bash
# Basic search
python pipeline.py --query "https://example.com/image.jpg"

# With options
python pipeline.py --query "image.jpg" --k 20 --gender "Women" --rebuild

# Force CPU usage
python pipeline.py --query "image.jpg" --device cpu
```

### Streamlit Web Interface

```bash
streamlit run app.py
```

Then open your browser to the provided local URL (typically http://localhost:8501).

## Dataset

This project uses the [Fashion Product Images Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset from Kaggle, which contains:
- ~44,000 product images
- Multiple categories (apparel, accessories, footwear, etc.)
- Gender annotations (Men, Women, Unisex, Boys, Girls)
- Rich metadata including product display names, colors, etc.

The dataset is automatically downloaded via [kagglehub](https://github.com/Kaggle/kagglehub) on first run.

## Requirements

- Python 3.7+
- Streamlit
- PyTorch
- Torchvision
- FAISS-CPU or FAISS-GPU
- Pillow
- Requests
- Kagglehub
- NumPy
- Pandas

See `requirements.txt` for specific versions.

## Architecture

```
image-search-engine/
├── app.py                 # Streamlit web application
├── pipeline.py            # Main pipeline orchestration
├── model.py               # ResNet50 feature extractor
├── dataset.py             # Dataset loading and preprocessing
├── indexer.py             # FAISS index management
├── search.py              # Search and display functionality
├── config.py              # Configuration constants
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Configuration

Modify `config.py` to adjust:
- `BATCH_SIZE`: DataLoader batch size (default: 128)
- `DEVICE`: "cuda" or "cpu" (auto-detected)
- `INDEX_TYPE`: FAISS index type (default: "IndexFlatIP" for cosine similarity)
- `VALID_GENDERS`: Allowed gender filter values
- `DEFAULT_TOP_K`: Default number of results (default: 10)

## Performance

- **First Run**: Downloads dataset (~1.5GB) and builds index (several minutes)
- **Subsequent Runs**: Loads pre-built index (seconds)
- **Search Speed**: Typically <100ms for k=10 on CPU, even faster on GPU

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [ResNet50](https://arxiv.org/abs/1512.03385) for feature extraction
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Streamlit](https://streamlit.io/) for the web interface
- [Kaggle](https://www.kaggle.com/) for the dataset

---

Built with ❤️ for computer vision and similarity search enthusiasts.