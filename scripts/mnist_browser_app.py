"""MNIST Browser - Streamlit web application for tagging and dataset management."""

import streamlit as st
import numpy as np
from pathlib import Path
import json
from typing import Optional, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.data_io import load_mnist_idx
from src.browser import TaggingSession, hash_image, image_to_base64, resize_image, get_image_stats
from src.browser.data_models import TaggedDataset


# ============================================================================
# PAGE CONFIG & SETUP
# ============================================================================

st.set_page_config(
    page_title="MNIST Browser & Tagger",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔢 MNIST Browser & Tagger")
st.markdown("Browse MNIST datasets, add custom tags, detect duplicates, and export tagged datasets.")


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def load_dataset(dataset_type: str, dataset_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Load MNIST dataset from specified path."""
    path_obj = Path(dataset_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    try:
        images, labels = load_mnist_idx(str(path_obj), kind="train")
        return images, labels, f"Loaded {len(images)} training images"
    except Exception as e:
        try:
            images, labels = load_mnist_idx(str(path_obj), kind="t10k")
            return images, labels, f"Loaded {len(images)} test images"
        except Exception as e2:
            raise ValueError(f"Could not load dataset: {str(e)} / {str(e2)}")


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "session" not in st.session_state:
        st.session_state.session = None
    if "current_image_idx" not in st.session_state:
        st.session_state.current_image_idx = 0
    if "images" not in st.session_state:
        st.session_state.images = None
    if "labels" not in st.session_state:
        st.session_state.labels = None
    if "dataset_source" not in st.session_state:
        st.session_state.dataset_source = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None
    if "existing_dataset" not in st.session_state:
        st.session_state.existing_dataset = None
    if "temp_tags" not in st.session_state:
        st.session_state.temp_tags = {}


initialize_session_state()


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.header("📁 Dataset Selection")
    
    dataset_type = st.radio(
        "Choose dataset source:",
        options=["Raw MNIST", "Load from File"],
        help="Select which MNIST dataset to browse"
    )
    
    if dataset_type == "Raw MNIST":
        dataset_path = str(project_root / "mnist_data")
        dataset_name = "raw_mnist"
    else:  # Load from File
        dataset_path = st.text_input(
            "Enter path to MNIST dataset folder:",
            placeholder="/path/to/mnist/data",
            help="Must contain train-images-idx3-ubyte and train-labels-idx1-ubyte"
        )
        dataset_name = "custom"
    
    # Load dataset button
    if st.button("📂 Load Dataset", use_container_width=True, key="load_dataset_btn"):
        if dataset_type == "Load from File" and not dataset_path:
            st.error("Please enter a dataset path")
        else:
            with st.spinner("Loading dataset..."):
                try:
                    images, labels, msg = load_dataset(dataset_type, dataset_path)
                    st.session_state.images = images
                    st.session_state.labels = labels
                    st.session_state.dataset_source = dataset_path
                    st.session_state.dataset_name = dataset_name
                    st.session_state.current_image_idx = 0
                    st.session_state.session = TaggingSession(
                        images, labels, source_dataset=dataset_name
                    )
                    st.success(msg)
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
    
    # Dataset info
    if st.session_state.images is not None:
        st.divider()
        st.subheader("📊 Dataset Info")
        st.metric("Total Images", len(st.session_state.images))
        st.metric("Image Size", f"{st.session_state.images.shape[1]}×{st.session_state.images.shape[2]}")
        st.metric("Unique Labels", len(np.unique(st.session_state.labels)) if st.session_state.labels is not None else "N/A")
    
    # Load existing tagged dataset
    st.divider()
    st.subheader("📋 Load Tagged Dataset")
    
    existing_file = st.file_uploader(
        "Upload JSON tagged dataset:",
        type="json",
        help="Load a previously saved tagged dataset for reference/duplicate checking"
    )
    
    if existing_file is not None:
        try:
            data = json.load(existing_file)
            st.session_state.existing_dataset = TaggedDataset(**data)
            st.success(f"Loaded {len(st.session_state.existing_dataset.images)} existing tags")
        except Exception as e:
            st.error(f"Error loading JSON: {str(e)}")
    
    # Export section
    st.divider()
    st.subheader("💾 Export Tagged Dataset")
    
    if st.session_state.session is not None:
        summary = st.session_state.session.get_summary()
        if summary['images_tagged'] > 0:
            st.info(f"✓ {summary['images_tagged']} images tagged")
            
            # Export filename
            export_name = st.text_input(
                "Export filename (without .json):",
                value="tagged_mnist",
                help="Filename for the exported JSON dataset"
            )
            
            if st.button("📤 Export to JSON", use_container_width=True, key="export_btn"):
                output_path = project_root / "data" / "tagged" / f"{export_name}.json"
                result = st.session_state.session.export_to_json(str(output_path))
                
                if result['success']:
                    st.success(result['message'])
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=f.read(),
                            file_name=f"{export_name}.json",
                            mime="application/json"
                        )
                else:
                    st.error(result['message'])
        else:
            st.warning("No images tagged yet. Add images first!")


# ============================================================================
# MAIN IMAGE VIEWER & TAGGING INTERFACE
# ============================================================================

if st.session_state.images is None:
    st.info("👈 Select and load a dataset from the sidebar to begin")
else:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Viewer", "🏷️ Batch Tagging", "📊 Dataset Summary"])
    
    # ========================================================================
    # TAB 1: IMAGE VIEWER
    # ========================================================================
    with tab1:
        st.subheader("Image Browser & Tagger")
        
        # Navigation controls
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.current_image_idx = max(0, st.session_state.current_image_idx - 1)
        
        with col2:
            new_idx = st.number_input(
                "Go to image:",
                min_value=0,
                max_value=len(st.session_state.images) - 1,
                value=st.session_state.current_image_idx,
                key="image_index"
            )
            st.session_state.current_image_idx = new_idx
        
        with col3:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.current_image_idx = min(
                    len(st.session_state.images) - 1,
                    st.session_state.current_image_idx + 1
                )
        
        with col4:
            st.metric("Position", f"{st.session_state.current_image_idx + 1} / {len(st.session_state.images)}")
        
        with col5:
            if st.button("🔄 Random", use_container_width=True):
                st.session_state.current_image_idx = np.random.randint(0, len(st.session_state.images))
        
        # Display current image
        current_idx = st.session_state.current_image_idx
        current_image = st.session_state.images[current_idx]
        current_label = st.session_state.labels[current_idx] if st.session_state.labels is not None else None
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display with resizing option
            display_size = st.slider("Display size:", 100, 400, 200, step=50)
            resized_image = resize_image(current_image, (display_size, display_size))
            st.image(resized_image, caption=f"Image #{current_idx}", use_container_width=False, width=display_size)
        
        with col2:
            st.subheader("📋 Image Details")
            
            # Image stats
            stats = get_image_stats(current_image)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Height", f"{stats['height']} px")
                st.metric("Width", f"{stats['width']} px")
                st.metric("Min Pixel", stats['min'])
            with col_b:
                st.metric("Max Pixel", stats['max'])
                st.metric("Mean", f"{stats['mean']:.1f}")
                st.metric("Std Dev", f"{stats['std']:.1f}")
            
            # Hash info
            current_hash = hash_image(current_image)
            st.text_area(
                "MD5 Hash (for deduplication):",
                value=current_hash,
                height=50,
                disabled=True
            )
            
            # Original label
            if current_label is not None:
                st.info(f"**Original Label:** {int(current_label)}")
            
            # Duplicate check
            st.divider()
            if st.button("⚠️ Check for Duplicates", use_container_width=True):
                candidates = st.session_state.session.get_duplicate_candidates(
                    current_idx,
                    st.session_state.existing_dataset
                )
                if candidates:
                    st.warning("**⚠️ Potential Duplicates Found!**")
                    for dup in candidates:
                        if dup['location'] == 'existing_dataset':
                            st.write(f"- In loaded dataset (ID: {dup['id']})")
                            st.write(f"  Tags: {', '.join(dup['tags']) or 'None'}")
                        else:
                            st.write(f"- In current session (Image #{dup['session_index']})")
                            st.write(f"  Tags: {', '.join(dup['tags']) or 'None'}")
                else:
                    st.success("✓ No duplicates found")
        
        # Tagging interface
        st.divider()
        st.subheader("🏷️ Add Tags")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            new_tag = st.text_input(
                "Enter tag name:",
                placeholder="e.g., US-style, rotated, crossed, italic",
                key=f"tag_input_{current_idx}"
            )
        
        with col2:
            if st.button("➕ Add Tag", use_container_width=True):
                if new_tag.strip():
                    image_in_session = any(
                        img.metadata.original_index == current_idx
                        for img in st.session_state.session.tagged_dataset.images
                    )

                    if image_in_session:
                        if st.session_state.session.add_tag(current_idx, new_tag.strip()):
                            st.success(f"✓ Added tag: '{new_tag.strip()}'")
                        else:
                            st.error("Could not add tag to existing image")
                    else:
                        result = st.session_state.session.add_image(
                            current_idx,
                            tags=[new_tag.strip()],
                            existing_dataset=st.session_state.existing_dataset
                        )
                        if result['success']:
                            if result['duplicate_found']:
                                st.warning(f"⚠️ Duplicate warning: {result['duplicate_info']['message']}")
                            st.success(f"✓ Added image and tag: '{new_tag.strip()}'")
                        else:
                            st.error(result['message'])
                    st.rerun()
        
        # Display current tags
        current_tagged_img = next(
            (img for img in st.session_state.session.tagged_dataset.images 
             if img.metadata.original_index == current_idx),
            None
        )
        
        if current_tagged_img:
            st.subheader("Current Tags")
            cols = st.columns(len(current_tagged_img.tags) + 1)
            for i, tag in enumerate(current_tagged_img.tags):
                with cols[i]:
                    if st.button(f"❌ {tag}", key=f"remove_{current_idx}_{tag}"):
                        if st.session_state.session.remove_tag(current_idx, tag):
                            st.success(f"Removed tag: '{tag}'")
                        else:
                            st.error(f"Could not remove tag: '{tag}'")
                        st.rerun()
        else:
            st.info("No tags added yet. Add your first tag above!")
        
        # Add current image to session (if not already added)
        if not current_tagged_img and st.button("📌 Add Image to Dataset", use_container_width=True):
            result = st.session_state.session.add_image(
                current_idx,
                existing_dataset=st.session_state.existing_dataset
            )
            if result['duplicate_found']:
                st.warning(f"⚠️ Duplicate warning: {result['duplicate_info']['message']}")
            st.success("Image added to tagged dataset!")
            st.rerun()
    
    # ========================================================================
    # TAB 2: BATCH TAGGING
    # ========================================================================
    with tab2:
        st.subheader("Batch Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Add Images by Range")
            start_idx = st.number_input(
                "Start index:",
                min_value=0,
                max_value=len(st.session_state.images) - 1,
                value=0
            )
            end_idx = st.number_input(
                "End index (inclusive):",
                min_value=0,
                max_value=len(st.session_state.images) - 1,
                value=min(99, len(st.session_state.images) - 1)
            )
            
            batch_tags = st.text_area(
                "Tags (one per line):",
                placeholder="tag1\ntag2\ntag3"
            )
            
            if st.button("📥 Add Range to Dataset", use_container_width=True):
                if start_idx <= end_idx:
                    tags_list = [t.strip() for t in batch_tags.split('\n') if t.strip()]
                    added_count = 0
                    for idx in range(start_idx, end_idx + 1):
                        result = st.session_state.session.add_image(
                            idx,
                            tags=tags_list,
                            existing_dataset=st.session_state.existing_dataset
                        )
                        if result['success']:
                            added_count += 1
                    
                    st.success(f"✓ Added {added_count} images with {len(tags_list)} tags each")
                    st.rerun()
        
        with col2:
            st.markdown("### Quick Stats")
            summary = st.session_state.session.get_summary()
            st.json(summary)
    
    # ========================================================================
    # TAB 3: DATASET SUMMARY
    # ========================================================================
    with tab3:
        st.subheader("Tagged Dataset Summary")
        
        summary = st.session_state.session.get_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images Available", summary['total_images_available'])
        with col2:
            st.metric("Images Tagged", summary['images_tagged'])
        with col3:
            st.metric("Unique Tags", summary['num_unique_tags'])
        with col4:
            st.metric("Avg Tags/Image", f"{summary['avg_tags_per_image']:.2f}")
        
        # Tag frequency
        if summary['unique_tags']:
            st.divider()
            st.subheader("Tag Usage")
            
            tag_counts = {}
            for img in st.session_state.session.tagged_dataset.images:
                for tag in img.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            if tag_counts:
                import pandas as pd
                df = pd.DataFrame(list(tag_counts.items()), columns=['Tag', 'Count'])
                df = df.sort_values('Count', ascending=False)
                
                st.bar_chart(df.set_index('Tag'))
                st.dataframe(df, use_container_width=True)
        
        # Duplicate warnings
        if st.session_state.session.duplicate_warnings:
            st.divider()
            st.subheader("⚠️ Duplicate Warnings")
            for warning in st.session_state.session.duplicate_warnings:
                st.warning(f"Image #{warning['duplicate_index']}: {warning['message']}")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
**MNIST Browser & Tagger** | Built with Streamlit  
📂 Data Path: `d:\\Developer\\DigitRecognizer\\`
""")
