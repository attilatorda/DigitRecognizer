# MNIST Browser & Tagger

A comprehensive Streamlit web application for browsing MNIST datasets, adding custom tags to images, detecting duplicates, and exporting tagged datasets in an API-compatible JSON format.

## Features

✅ **Browse MNIST Datasets**
- Load raw MNIST (60,000 training images)
- Load custom MNIST datasets
- Navigate with next/prev/jump/random controls

✅ **Custom Multi-Label Tagging**
- Add unlimited tags per image
- Remove tags easily
- Support for any tag name (e.g., "US-style", "rotated", "italic", "crossed")

✅ **Duplicate Detection**
- MD5-based hash detection for exact duplicates
- Warns before adding duplicate images
- Detects duplicates across sessions and datasets

✅ **Image Analysis**
- Display images with adjustable size (100-400px)
- Show image statistics (height, width, min/max/mean/std pixel values)
- Display MD5 hash for each image
- Show original MNIST label

✅ **API-Compatible Export Format**
- Export to JSON with full pixel data (base64-encoded)
- Multiple labels per image supported
- Rich metadata (source dataset, original label, user notes, timestamps)
- Compatible with REST APIs and data pipelines

✅ **Batch Operations**
- Add multiple images by range
- Apply same tags to batch
- Session persistence with export/import

✅ **Dataset Statistics**
- Track tagged images count
- Unique tags analysis
- Tag frequency charts
- Duplicate warning tracking

## Installation

1. **Install dependencies:**
   ```bash
   cd d:\Developer\DigitRecognizer
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python scripts/test_mnist_browser.py
   ```

   Expected output:
   ```
   ✓ ALL TESTS PASSED!
   ```

## Quick Start

### Launch the App

```bash
cd d:\Developer\DigitRecognizer
streamlit run scripts/mnist_browser_app.py
```

The app will open in your browser at `http://localhost:8501`

### Typical Workflow

1. **Load a Dataset**
   - In the sidebar, select "Raw MNIST" or load from a custom folder
   - Click "📂 Load Dataset"
   - Wait for the dataset to load (~2-3 seconds)

2. **Browse and Tag Images**
   - Use the navigation controls (Previous/Next/Go to/Random) to browse
   - View image details (size, hash, pixel statistics)
   - Add tags in the "🏷️ Add Tags" section (e.g., "US-style", "rotated")
   - Click "➕ Add Tag" to apply

3. **Check for Duplicates**
   - Click "⚠️ Check for Duplicates" to scan for existing instances
   - Load a reference dataset in the sidebar first to cross-reference

4. **Export Your Annotations**
   - In the sidebar, enter an export filename
   - Click "📤 Export to JSON"
   - Download the resulting JSON file

5. **Load and Continue**
   - Upload a previously saved JSON file in the sidebar
   - Continue editing tags and export again

## JSON Format Specification

### Schema Overview

```json
{
  "metadata": {
    "version": "1.0",
    "created_at": "2026-05-22T12:00:00",
    "total_images": 150,
    "image_height": 28,
    "image_width": 28,
    "duplicate_count": 3,
    "description": "Custom MNIST variant with style labels"
  },
  "images": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "md5_hash": "5d41402abc4b2a76b9719d911017c592",
      "pixel_data": "iVBORw0KGgoAAAANSUhEUgAAABw...",
      "tags": ["US-style", "rotated"],
      "metadata": {
        "source_dataset": "raw_mnist",
        "original_label": 7,
        "original_index": 1000,
        "notes": "Crossed seven with unique styling"
      }
    }
  ]
}
```

### Field Descriptions

**Dataset Metadata:**
- `version`: Format version (currently "1.0")
- `created_at`: ISO-8601 timestamp
- `total_images`: Count of images in dataset
- `image_height`, `image_width`: Dimensions in pixels
- `duplicate_count`: Number of duplicates detected during tagging
- `description`: Optional user description

**Per-Image Data:**
- `id`: Unique UUID for each image
- `md5_hash`: MD5 hash of pixel data (for deduplication)
- `pixel_data`: Base64-encoded PNG image bytes
- `tags`: Array of custom labels/tags
- `metadata.source_dataset`: Origin ("raw_mnist", "custom", etc.)
- `metadata.original_label`: Original MNIST digit (0-9)
- `metadata.original_index`: Index in source dataset
- `metadata.notes`: Optional user annotations

## API Usage

### Load a Tagged Dataset

```python
import json
from src.browser import TaggingSession

with open("tagged_mnist.json", "r") as f:
    data = json.load(f)

session, msg = TaggingSession.load_from_json("tagged_mnist.json")
print(f"Loaded {len(session.tagged_dataset.images)} images")

# Access images
for img in session.tagged_dataset.images:
    print(f"ID: {img.id}, Tags: {img.tags}")
```

### Convert Base64 Back to Images

```python
import numpy as np
from src.browser import base64_to_image
import json

with open("tagged_mnist.json", "r") as f:
    data = json.load(f)

for img_data in data["images"]:
    pixel_array = base64_to_image(img_data["pixel_data"])
    print(f"Image shape: {pixel_array.shape}")
    # Use pixel_array for ML pipelines, visualization, etc.
```

### Query by Tags

```python
import json

with open("tagged_mnist.json", "r") as f:
    data = json.load(f)

# Find all "rotated" digits
rotated_images = [
    img for img in data["images"]
    if "rotated" in img["tags"]
]
print(f"Found {len(rotated_images)} rotated images")
```

## Project Structure

```
src/browser/
├── __init__.py              # Package exports
├── data_models.py           # Pydantic models (TaggedImage, TaggedDataset, etc.)
├── image_utils.py           # Image operations (hash, base64, resize)
└── tagging_engine.py        # TaggingSession class

scripts/
├── mnist_browser_app.py     # Main Streamlit application
└── test_mnist_browser.py    # Test suite

data/
└── tagged/                  # Output directory for exported JSON files
```

## Core Classes & Functions

### TaggingSession

Main class for managing a tagging workflow.

```python
session = TaggingSession(images, labels, source_dataset="raw_mnist")

# Add an image with tags
result = session.add_image(0, tags=["US-style", "rotated"])

# Check for duplicates
candidates = session.get_duplicate_candidates(0, existing_dataset=other_dataset)

# Export to JSON
session.export_to_json("output.json")

# Load from JSON
session, msg = TaggingSession.load_from_json("output.json")

# Get summary statistics
summary = session.get_summary()
```

### Image Utilities

```python
from src.browser import hash_image, image_to_base64, base64_to_image, resize_image

# Hash image for deduplication
img_hash = hash_image(image_array)

# Encode to base64 for JSON storage
encoded = image_to_base64(image_array)

# Decode from base64
decoded = base64_to_image(encoded)

# Resize for display
resized = resize_image(image_array, (224, 224))
```

### Data Models

```python
from src.browser import TaggedImage, TaggedDataset, ImageMetadata

# All are Pydantic BaseModel classes
# Full validation and serialization support
# JSON schema auto-generation
```

## Use Cases

### 1. Create Style-Annotated MNIST Dataset
```
Label digits with handwriting styles: "US-style", "European", "artistic", "formal", etc.
→ Export → Use for style-transfer research
```

### 2. Crowdsource Annotations
```
Share the app, collect tags from multiple annotators
→ Each exports JSON
→ Merge JSONs → Analyze tag consensus
```

### 3. Data Pipeline Integration
```
Load JSON in Python script
→ Query by tags
→ Serve selected images to ML model
→ No need to re-annotate on model retraining
```

### 4. Quality Assurance
```
Tag "unclear", "damaged", "mislabeled" images
→ Filter them out from training data
→ Improve model quality
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Run `pip install -r requirements.txt` to install all dependencies.

### Issue: "FileNotFoundError: mnist_data not found"
**Solution:** Ensure the `mnist_data/` folder exists in `d:\Developer\DigitRecognizer\`. If not, update the path in the sidebar.

### Issue: Streamlit app is slow
**Solution:** The app caches datasets after first load. Subsequent browses are instant. For very large custom datasets, consider loading a subset first.

### Issue: JSON file too large
**Solution:** This is expected - base64-encoded 28×28 MNIST images are ~750 bytes each. 10,000 images ≈ 7.5 MB JSON. For large-scale use, store images in a database and reference by ID instead.

## Future Enhancements

- [ ] Support for image formats beyond MNIST (CIFAR-10, ImageNet, etc.)
- [ ] Collaborative tagging with user authentication
- [ ] Tag suggestions based on image content (ML-powered)
- [ ] Advanced duplicate detection (perceptual hashing)
- [ ] Bulk tag operations (rename, merge, delete)
- [ ] Export to other formats (CSV, COCO, YOLO)
- [ ] Image augmentation preview
- [ ] Integration with active learning pipelines

## License

Same as parent project (DigitRecognizer)

## Contributing

For improvements or bug reports, please refer to the main project repository.

---

**Questions?** Check the test suite (`scripts/test_mnist_browser.py`) for working examples.
