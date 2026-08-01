import streamlit as st
import sys
import os
from PIL import Image
import requests
from io import BytesIO

# Add the current directory to the path so we can import the pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import ImageSearchPipeline

# Set page config
st.set_page_config(
    page_title="Image Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔍 Image Search Engine")
st.markdown("""
Search for similar images using deep learning features and FAISS similarity search.
Upload an image or provide an image URL to find similar products.
""")

# Sidebar for options
st.sidebar.header("Search Options")

# Input method selection
input_method = st.sidebar.radio(
    "Select input method:",
    ("Upload Image", "Image URL")
)

# Initialize variables
query_input = None
uploaded_file = None

if input_method == "Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
        # Convert to bytes for pipeline
        query_input = uploaded_file
else:
    query_input = st.sidebar.text_input(
        "Enter image URL:",
        placeholder="https://example.com/image.jpg"
    )

# Other options
k = st.sidebar.slider("Number of results:", min_value=1, max_value=20, value=10)
gender_options = ["Men", "Women", "Unisex", "Boys", "Girls", None]
gender_labels = ["Men", "Women", "Unisex", "Boys", "Girls", "No filter"]
gender_index = st.sidebar.selectbox(
    "Filter by gender:",
    options=range(len(gender_options)),
    format_func=lambda x: gender_labels[x],
    index=len(gender_options)-1  # Default to "No filter"
)
filter_gender = gender_options[gender_index] if gender_options[gender_index] is not None else None

force_rebuild = st.sidebar.checkbox("Force rebuild index", value=False)

# Search button
if st.sidebar.button("Search", type="primary"):
    if input_method == "Upload Image" and uploaded_file is None:
        st.sidebar.error("Please upload an image")
    elif input_method == "Image URL" and not query_input:
        st.sidebar.error("Please enter an image URL")
    else:
        # Create pipeline and run search
        with st.spinner("Initializing search pipeline..."):
            try:
                pipeline = ImageSearchPipeline()
                
                # Setup components if needed
                if force_rebuild or not hasattr(pipeline, 'index') or pipeline.index is None:
                    with st.spinner("Setting up dataset..."):
                        pipeline.setup_dataset()
                    with st.spinner("Loading model..."):
                        pipeline.setup_model()
                    with st.spinner("Building index..."):
                        pipeline.build_index()
                
                # Perform search
                with st.spinner("Searching for similar images..."):
                    results = pipeline.search(
                        query_input=query_input,
                        k=k,
                        filter_gender=filter_gender,
                        force_rebuild=force_rebuild
                    )
                
                # Display results
                if results:
                    st.success(f"Found {len(results)} similar images!")
                    
                    # Display results in a grid
                    cols = st.columns(5)
                    for idx, (img_path, score, metadata) in enumerate(results):
                        col = cols[idx % 5]
                        with col:
                            try:
                                # Try to load and display image
                                                                if isinstance(img_path, str) and img_path.startswith('http'):
                                                                    response = requests.get(img_path)
                                                                    img = Image.open(BytesIO(response.content))
                                                                else:
                                                                    img = Image.open(img_path)
                                img.thumbnail((200, 200))
                                st.image(img, use_column_width=True)
                                caption = f"Score: {score:.3f}"
                                if metadata and 'productDisplayName' in metadata:
                                    caption += f"<br>{metadata['productDisplayName']}"
                                st.markdown(caption, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
                else:
                    st.warning("No results found. Try a different image or adjust filters.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your internet connection and try again.")

# Instructions
with st.expander("How to use"):
    st.markdown("""
    1. Choose whether to upload an image or provide an image URL
    2. Select the number of similar images you want to see (1-20)
    3. Optionally filter by gender category
    4. Click 'Search' to find similar images
    5. Results will be displayed in a grid below
    
    **Notes:**
    - First run may take several minutes as it downloads the dataset and builds the index
    - Subsequent searches will be much faster as the index is saved to disk
    - The system uses ResNet50 features and FAISS for efficient similarity search
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, ResNet50, and FAISS")