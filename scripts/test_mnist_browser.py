"""Quick test of MNIST browser functionality."""

import sys
import numpy as np
from pathlib import Path
import json
import tempfile

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.data_io import load_mnist_idx
from src.browser import TaggingSession, hash_image, image_to_base64, base64_to_image

def test_mnist_browser():
    """Test core MNIST browser functionality."""
    
    print("\n" + "="*60)
    print("MNIST BROWSER TEST SUITE")
    print("="*60)
    
    # Test 1: Load MNIST data
    print("\n[1/8] Testing MNIST data loading...")
    try:
        mnist_path = project_root / "mnist_data"
        images, labels = load_mnist_idx(str(mnist_path), kind="train")
        print(f"  ✓ Loaded {len(images)} images")
        print(f"  ✓ Image shape: {images[0].shape}")
        print(f"  ✓ Labels: {np.unique(labels)}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 2: Create tagging session
    print("\n[2/8] Testing TaggingSession creation...")
    try:
        session = TaggingSession(images[:100], labels[:100], source_dataset="test_mnist")
        print(f"  ✓ Session created with {session.num_images} images")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 3: Hash image
    print("\n[3/8] Testing image hashing...")
    try:
        img_hash = hash_image(images[0])
        print(f"  ✓ Hash for image 0: {img_hash[:16]}...")
        # Same image should have same hash
        img_hash2 = hash_image(images[0])
        assert img_hash == img_hash2, "Hash not consistent!"
        print(f"  ✓ Hash consistency verified")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 4: Base64 encoding/decoding
    print("\n[4/8] Testing base64 encoding/decoding...")
    try:
        original = images[0]
        encoded = image_to_base64(original)
        print(f"  ✓ Encoded image (base64 length: {len(encoded)})")
        decoded = base64_to_image(encoded)
        assert np.array_equal(original, decoded), "Decoded image doesn't match original!"
        print(f"  ✓ Roundtrip successful")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 5: Add images and tags
    print("\n[5/8] Testing add_image and tagging...")
    try:
        # Add first image with tags
        result1 = session.add_image(0, tags=["US-style", "rotated"])
        assert result1['success'], "Failed to add image 0"
        print(f"  ✓ Added image 0 with 2 tags")
        
        # Add second image
        result2 = session.add_image(1, tags=["italic"])
        assert result2['success'], "Failed to add image 1"
        print(f"  ✓ Added image 1 with 1 tag")
        
        # Check session state
        summary = session.get_summary()
        assert summary['images_tagged'] == 2, "Expected 2 images tagged"
        assert summary['num_unique_tags'] == 3, "Expected 3 unique tags"
        print(f"  ✓ Session has {summary['images_tagged']} images, {summary['num_unique_tags']} unique tags")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 6: Re-adding same image index should merge tags, not duplicate entries
    print("\n[6/8] Testing merge behavior when re-adding same image...")
    try:
        before_count = len(session.tagged_dataset.images)
        merge_result = session.add_image(0, tags=["crossed", "US-style"])
        after_count = len(session.tagged_dataset.images)
        assert merge_result['success'], "Merge add_image should succeed"
        assert before_count == after_count, "Re-adding same index created duplicate entry"
        img0 = next(img for img in session.tagged_dataset.images if img.metadata.original_index == 0)
        assert "crossed" in img0.tags, "Merged tag not present"
        assert img0.tags.count("US-style") == 1, "Duplicate tag introduced during merge"
        print("  ✓ Re-adding same index merges tags without duplicate rows")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # Test 7: add_tag/remove_tag should work with sparse source indices
    print("\n[7/8] Testing add_tag/remove_tag with sparse source indices...")
    try:
        sparse_session = TaggingSession(images[:200], labels[:200], source_dataset="sparse_test")
        add_sparse = sparse_session.add_image(100, tags=["baseline"])
        assert add_sparse['success'], "Failed to add sparse index image"
        assert sparse_session.add_tag(100, "italic"), "add_tag failed for sparse index"
        sparse_img = next(img for img in sparse_session.tagged_dataset.images if img.metadata.original_index == 100)
        assert "italic" in sparse_img.tags, "Tag not added for sparse index"
        assert sparse_session.remove_tag(100, "baseline"), "remove_tag failed for sparse index"
        sparse_img = next(img for img in sparse_session.tagged_dataset.images if img.metadata.original_index == 100)
        assert "baseline" not in sparse_img.tags, "Tag not removed for sparse index"
        print("  ✓ add_tag/remove_tag works correctly for sparse indices")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # Test 8: Export and reload JSON
    print("\n[8/8] Testing JSON export/reload...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "test_export.json"
            
            # Export
            result = session.export_to_json(str(export_path))
            assert result['success'], f"Export failed: {result['message']}"
            print(f"  ✓ Exported to JSON (file size: {result['file_size']} bytes)")
            
            # Load and verify
            loaded_session, msg = TaggingSession.load_from_json(str(export_path))
            assert loaded_session is not None, f"Failed to load: {msg}"
            assert len(loaded_session.tagged_dataset.images) == 2, "Expected 2 images after reload"
            print(f"  ✓ Reloaded {len(loaded_session.tagged_dataset.images)} images from JSON")
            
            # Verify content
            img0 = loaded_session.tagged_dataset.images[0]
            assert "US-style" in img0.tags, "Lost tag: US-style"
            assert "rotated" in img0.tags, "Lost tag: rotated"
            print(f"  ✓ Tags preserved: {img0.tags}")
            
            # Read JSON and check schema
            with open(export_path, 'r') as f:
                data = json.load(f)
            assert 'metadata' in data, "Missing 'metadata' in JSON"
            assert 'images' in data, "Missing 'images' in JSON"
            assert data['metadata']['version'] == "1.0", "Wrong version"
            print(f"  ✓ JSON schema valid")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60 + "\n")
    return True


if __name__ == "__main__":
    success = test_mnist_browser()
    sys.exit(0 if success else 1)
