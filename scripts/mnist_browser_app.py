"""MNIST Browser - Streamlit web application for tagging and dataset management."""

import streamlit as st
import numpy as np
from pathlib import Path
import json
from typing import Optional, Tuple
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.data_io import load_mnist_idx
from src.browser import TaggingSession, hash_image, resize_image, get_image_stats
from src.browser.data_models import TaggedDataset


# ============================================================================
# PAGE CONFIG & SETUP
# ============================================================================

st.set_page_config(
    page_title="MNIST Browser",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MNIST Browser")
st.markdown("Browse datasets, add custom tags, detect duplicates, and export tagged datasets.")


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
        return images, labels, f"Loaded {len(images)} training images ({dataset_type})"
    except Exception as e:
        try:
            images, labels = load_mnist_idx(str(path_obj), kind="t10k")
            return images, labels, f"Loaded {len(images)} test images ({dataset_type})"
        except Exception as e2:
            raise ValueError(f"Could not load dataset: {str(e)} / {str(e2)}")


def _detect_cultivar17(labels: Optional[np.ndarray]) -> bool:
    """Return True if labels look like CultiVar-17 class indices (0-16, exactly 17 unique)."""
    if labels is None:
        return False
    unique = np.unique(labels)
    return len(unique) == 17 and int(unique.max()) == 16


def initialize_session_state():
    defaults = {
        "session": None,
        "current_image_idx": 0,
        "image_index": 0,
        "images": None,
        "labels": None,
        "dataset_source": None,
        "dataset_name": None,
        "existing_dataset": None,
        "temp_tags": {},
        "gallery_page": 0,
        "gallery_page_size": 20,
        "is_cultivar17": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


initialize_session_state()


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.header("Dataset Selection")

    dataset_type = st.radio(
        "Choose dataset source:",
        options=["Raw MNIST", "Load from File", "Load .npy file"],
        help="Select which dataset to browse",
    )

    if dataset_type == "Raw MNIST":
        dataset_path = str(project_root / "mnist_data")
        dataset_name = "raw_mnist"
    elif dataset_type == "Load from File":
        dataset_path = st.text_input(
            "Path to MNIST dataset folder:",
            placeholder="/path/to/mnist/data",
            help="Must contain train-images-idx3-ubyte and train-labels-idx1-ubyte",
        )
        dataset_name = "custom"
    else:
        dataset_path = None
        dataset_name = "npy"

    npy_images_file = None
    npy_labels_file = None
    if dataset_type == "Load .npy file":
        npy_images_file = st.file_uploader(
            "Images .npy  (N×H×W or H×W):",
            type=["npy"],
            help="Float32 [0,1] or uint8 [0,255]",
        )
        npy_labels_file = st.file_uploader(
            "Labels .npy  (optional):",
            type=["npy"],
            help="Integer array of shape (N,) matching the images",
        )

    if st.button("Load Dataset", use_container_width=True, key="load_dataset_btn"):
        if dataset_type == "Load from File" and not dataset_path:
            st.error("Please enter a dataset path")
        elif dataset_type == "Load .npy file" and npy_images_file is None:
            st.error("Please upload a .npy images file")
        else:
            with st.spinner("Loading..."):
                try:
                    if dataset_type == "Load .npy file":
                        raw = np.load(npy_images_file)
                        if raw.ndim == 2:
                            raw = raw[np.newaxis, ...]
                        if raw.ndim != 3:
                            raise ValueError(f"Expected 2D or 3D array, got shape {raw.shape}")
                        if raw.dtype != np.uint8:
                            if raw.max() <= 1.0:
                                raw = (raw * 255).clip(0, 255).astype(np.uint8)
                            else:
                                raw = raw.astype(np.uint8)
                        images = raw
                        labels = np.load(npy_labels_file) if npy_labels_file is not None else None
                        file_name = npy_images_file.name
                        msg = f"Loaded {len(images)} images ({images.shape[1]}x{images.shape[2]} px)"
                        dataset_name = file_name
                        source = file_name
                    else:
                        images, labels, msg = load_dataset(dataset_type, str(dataset_path))
                        source = dataset_path or ""

                    st.session_state.images = images
                    st.session_state.labels = labels
                    st.session_state.dataset_source = source
                    st.session_state.dataset_name = dataset_name
                    st.session_state.current_image_idx = 0
                    st.session_state["image_index"] = 0
                    st.session_state.is_cultivar17 = _detect_cultivar17(labels)
                    st.session_state.session = TaggingSession(
                        images, labels, source_dataset=dataset_name
                    )
                    st.success(msg)
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")

    if st.session_state.images is not None:
        st.divider()
        st.subheader("Dataset Info")
        st.metric("Total Images", len(st.session_state.images))
        st.metric("Image Size", f"{st.session_state.images.shape[1]}x{st.session_state.images.shape[2]}")
        st.metric(
            "Unique Labels",
            len(np.unique(st.session_state.labels)) if st.session_state.labels is not None else "N/A",
        )

    st.divider()
    st.subheader("Load Tagged Dataset")

    existing_file = st.file_uploader(
        "Upload JSON tagged dataset:",
        type="json",
        help="Load a previously saved tagged dataset for reference/duplicate checking",
    )
    if existing_file is not None:
        try:
            data = json.load(existing_file)
            st.session_state.existing_dataset = TaggedDataset(**data)
            st.success(f"Loaded {len(st.session_state.existing_dataset.images)} existing tags")
        except Exception as e:
            st.error(f"Error loading JSON: {e}")

    st.divider()
    st.subheader("Export Tagged Dataset")

    if st.session_state.session is not None:
        summary = st.session_state.session.get_summary()
        if summary["images_tagged"] > 0:
            st.info(f"{summary['images_tagged']} images tagged")
            export_name = st.text_input(
                "Export filename (without .json):",
                value="tagged_mnist",
            )
            if st.button("Export to JSON", use_container_width=True, key="export_btn"):
                output_path = project_root / "data" / "tagged" / f"{export_name}.json"
                result = st.session_state.session.export_to_json(str(output_path))
                if result["success"]:
                    st.success(result["message"])
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download JSON",
                            data=f.read(),
                            file_name=f"{export_name}.json",
                            mime="application/json",
                        )
                else:
                    st.error(result["message"])
        else:
            st.warning("No images tagged yet.")


# ============================================================================
# MAIN IMAGE VIEWER & TAGGING INTERFACE
# ============================================================================

if st.session_state.images is None:
    st.info("Select and load a dataset from the sidebar to begin.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Image Viewer", "Batch Tagging", "Summary", "Gallery"])

    # ========================================================================
    # TAB 1: IMAGE VIEWER
    # ========================================================================
    with tab1:
        N = len(st.session_state.images)

        # Navigation controls — bottom-aligned so buttons sit level with the input.
        # Buttons are rendered (and their click state read) BEFORE the number_input
        # so that session state can be updated before the keyed widget is instantiated.
        col1, col2, col3, col4, col5 = st.columns(5, vertical_alignment="bottom")

        with col1:
            prev_clicked = st.button("< Previous", use_container_width=True)
        with col3:
            next_clicked = st.button("Next >", use_container_width=True)
        with col5:
            rand_clicked = st.button("Random", use_container_width=True)

        # Apply navigation before the number_input is instantiated
        if prev_clicked:
            st.session_state["image_index"] = max(0, st.session_state["image_index"] - 1)
        elif next_clicked:
            st.session_state["image_index"] = min(N - 1, st.session_state["image_index"] + 1)
        elif rand_clicked:
            st.session_state["image_index"] = int(np.random.randint(0, N))

        with col2:
            # No value= parameter — widget state is owned entirely by the key
            new_idx = st.number_input(
                "Go to image:",
                min_value=0,
                max_value=N - 1,
                key="image_index",
            )

        st.session_state.current_image_idx = int(st.session_state["image_index"])

        with col4:
            st.metric("Position", f"{st.session_state.current_image_idx + 1} / {N}")

        # Display current image
        current_idx = st.session_state.current_image_idx
        current_image = st.session_state.images[current_idx]
        current_label = (
            st.session_state.labels[current_idx]
            if st.session_state.labels is not None
            else None
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            display_size = st.slider("Display size:", 100, 400, 200, step=50)
            resized_image = resize_image(current_image, (display_size, display_size))
            st.image(resized_image, caption=f"Image #{current_idx}", width=display_size)

        with col2:
            st.subheader("Image Details")

            stats = get_image_stats(current_image)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Height", f"{stats['height']} px")
                st.metric("Width", f"{stats['width']} px")
                st.metric("Min Pixel", stats["min"])
            with col_b:
                st.metric("Max Pixel", stats["max"])
                st.metric("Mean", f"{stats['mean']:.1f}")
                st.metric("Std Dev", f"{stats['std']:.1f}")

            current_hash = hash_image(current_image)
            st.text_area("MD5 Hash:", value=current_hash, height=50, disabled=True)

            # Label — show digit + variant name for CultiVar-17, raw value otherwise
            if current_label is not None:
                raw = int(current_label)
                if st.session_state.get("is_cultivar17") and 0 <= raw <= 16:
                    try:
                        from src.variants17.label_schema import CLASS17_TO_DIGIT10, LABELS_17
                        digit = CLASS17_TO_DIGIT10[raw]
                        variant = LABELS_17[raw]
                        st.info(f"Label: **{digit}**  ({variant})")
                    except Exception:
                        st.info(f"Label: {raw}")
                else:
                    st.info(f"Label: **{raw}**")

            st.divider()
            if st.button("Check for Duplicates", use_container_width=True):
                candidates = st.session_state.session.get_duplicate_candidates(
                    current_idx, st.session_state.existing_dataset
                )
                if candidates:
                    st.warning("Potential duplicates found")
                    for dup in candidates:
                        if dup["location"] == "existing_dataset":
                            st.write(f"- In loaded dataset (ID: {dup['id']})")
                            st.write(f"  Tags: {', '.join(dup['tags']) or 'None'}")
                        else:
                            st.write(f"- In current session (Image #{dup['session_index']})")
                            st.write(f"  Tags: {', '.join(dup['tags']) or 'None'}")
                else:
                    st.success("No duplicates found")

        # Tagging interface
        st.divider()
        st.subheader("Add Tags")

        col1, col2 = st.columns([2, 1], vertical_alignment="bottom")
        with col1:
            new_tag = st.text_input(
                "Tag:",
                placeholder="e.g., crossed, italic, ambiguous",
                key=f"tag_input_{current_idx}",
            )
        with col2:
            if st.button("Add Tag", use_container_width=True):
                if new_tag.strip():
                    image_in_session = any(
                        img.metadata.original_index == current_idx
                        for img in st.session_state.session.tagged_dataset.images
                    )
                    if image_in_session:
                        if st.session_state.session.add_tag(current_idx, new_tag.strip()):
                            st.success(f"Added tag: '{new_tag.strip()}'")
                        else:
                            st.error("Could not add tag")
                    else:
                        result = st.session_state.session.add_image(
                            current_idx,
                            tags=[new_tag.strip()],
                            existing_dataset=st.session_state.existing_dataset,
                        )
                        if result["success"]:
                            if result["duplicate_found"]:
                                st.warning(f"Duplicate warning: {result['duplicate_info']['message']}")
                            st.success(f"Added tag: '{new_tag.strip()}'")
                        else:
                            st.error(result["message"])
                    st.rerun()

        current_tagged_img = next(
            (
                img
                for img in st.session_state.session.tagged_dataset.images
                if img.metadata.original_index == current_idx
            ),
            None,
        )

        if current_tagged_img:
            st.subheader("Current Tags")
            cols = st.columns(len(current_tagged_img.tags) + 1)
            for i, tag in enumerate(current_tagged_img.tags):
                with cols[i]:
                    if st.button(f"[x] {tag}", key=f"remove_{current_idx}_{tag}"):
                        if st.session_state.session.remove_tag(current_idx, tag):
                            st.success(f"Removed tag: '{tag}'")
                        else:
                            st.error(f"Could not remove tag: '{tag}'")
                        st.rerun()
        else:
            st.caption("No tags added yet.")

        if not current_tagged_img and st.button("Add Image to Dataset", use_container_width=True):
            result = st.session_state.session.add_image(
                current_idx, existing_dataset=st.session_state.existing_dataset
            )
            if result["duplicate_found"]:
                st.warning(f"Duplicate warning: {result['duplicate_info']['message']}")
            st.success("Image added to dataset.")
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
                value=0,
            )
            end_idx = st.number_input(
                "End index (inclusive):",
                min_value=0,
                max_value=len(st.session_state.images) - 1,
                value=min(99, len(st.session_state.images) - 1),
            )
            batch_tags = st.text_area("Tags (one per line):", placeholder="tag1\ntag2")

            if st.button("Add Range to Dataset", use_container_width=True):
                if start_idx <= end_idx:
                    tags_list = [t.strip() for t in batch_tags.split("\n") if t.strip()]
                    added_count = 0
                    for idx in range(start_idx, end_idx + 1):
                        result = st.session_state.session.add_image(
                            idx,
                            tags=tags_list,
                            existing_dataset=st.session_state.existing_dataset,
                        )
                        if result["success"]:
                            added_count += 1
                    st.success(f"Added {added_count} images with {len(tags_list)} tags each")
                    st.rerun()

        with col2:
            st.markdown("### Quick Stats")
            summary = st.session_state.session.get_summary()
            st.json(summary)

    # ========================================================================
    # TAB 3: SUMMARY
    # ========================================================================
    with tab3:
        st.subheader("Tagged Dataset Summary")

        summary = st.session_state.session.get_summary()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", summary["total_images_available"])
        with col2:
            st.metric("Images Tagged", summary["images_tagged"])
        with col3:
            st.metric("Unique Tags", summary["num_unique_tags"])
        with col4:
            st.metric("Avg Tags/Image", f"{summary['avg_tags_per_image']:.2f}")

        if summary["unique_tags"]:
            st.divider()
            st.subheader("Tag Usage")
            tag_counts = {}
            for img in st.session_state.session.tagged_dataset.images:
                for tag in img.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if tag_counts:
                import pandas as pd
                df = pd.DataFrame(list(tag_counts.items()), columns=["Tag", "Count"])
                df = df.sort_values("Count", ascending=False)
                st.bar_chart(df.set_index("Tag"))
                st.dataframe(df, use_container_width=True)

        if st.session_state.session.duplicate_warnings:
            st.divider()
            st.subheader("Duplicate Warnings")
            for warning in st.session_state.session.duplicate_warnings:
                st.warning(f"Image #{warning['duplicate_index']}: {warning['message']}")

    # ========================================================================
    # TAB 4: GALLERY
    # ========================================================================
    with tab4:
        st.subheader("Gallery")

        N = len(st.session_state.images)

        gcol1, gcol2, gcol3, gcol4 = st.columns([2, 1, 1, 2], vertical_alignment="bottom")
        with gcol1:
            new_page_size = st.selectbox(
                "Images per page:",
                options=[10, 20, 50],
                index=[10, 20, 50].index(st.session_state.gallery_page_size),
                key="gallery_page_size_select",
            )
            if new_page_size != st.session_state.gallery_page_size:
                st.session_state.gallery_page_size = new_page_size
                st.session_state.gallery_page = 0
                st.rerun()

        page_size = st.session_state.gallery_page_size
        total_pages = max(1, (N + page_size - 1) // page_size)
        page = min(st.session_state.gallery_page, total_pages - 1)

        with gcol2:
            if st.button("< Prev", disabled=(page == 0), key="gallery_prev"):
                st.session_state.gallery_page = page - 1
                st.rerun()
        with gcol3:
            if st.button("Next >", disabled=(page >= total_pages - 1), key="gallery_next"):
                st.session_state.gallery_page = page + 1
                st.rerun()
        with gcol4:
            start = page * page_size
            end = min(start + page_size, N)
            st.caption(f"Page {page + 1} / {total_pages}  |  {start + 1}-{end} of {N}")

        st.divider()

        tagged_indices: set = set()
        if st.session_state.session is not None:
            for img in st.session_state.session.tagged_dataset.images:
                if img.metadata.original_index is not None:
                    tagged_indices.add(img.metadata.original_index)

        COLS = 5
        cols = st.columns(COLS)
        for i, idx in enumerate(range(start, end)):
            with cols[i % COLS]:
                thumb = resize_image(st.session_state.images[idx], (112, 112))
                st.image(thumb, width=112)

                # Build label string
                lbl = st.session_state.labels[idx] if st.session_state.labels is not None else None
                if lbl is not None:
                    raw = int(lbl)
                    if st.session_state.get("is_cultivar17") and 0 <= raw <= 16:
                        try:
                            from src.variants17.label_schema import CLASS17_TO_DIGIT10, LABELS_17
                            label_str = str(CLASS17_TO_DIGIT10[raw])
                        except Exception:
                            label_str = str(raw)
                    else:
                        label_str = str(raw)
                else:
                    label_str = "?"

                tag_marker = " *" if idx in tagged_indices else ""
                if st.button(
                    f"#{idx} [{label_str}]{tag_marker}",
                    key=f"gal_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.current_image_idx = idx
                    st.session_state["image_index"] = idx
                    st.info(f"Jumped to image #{idx} — switch to Image Viewer tab")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(f"MNIST Browser  |  Data Path: {project_root}")
