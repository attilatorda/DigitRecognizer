"""Image utility functions for MNIST browser."""

import numpy as np
import hashlib
import base64
from typing import Tuple
from io import BytesIO
from PIL import Image


def hash_image(image_array: np.ndarray) -> str:
    """
    Compute MD5 hash of image array for duplicate detection.
    
    Args:
        image_array: 2D numpy array of pixel values (H x W)
    
    Returns:
        MD5 hash string
    """
    # Convert to bytes and hash
    image_bytes = image_array.tobytes()
    return hashlib.md5(image_bytes).hexdigest()


def image_to_base64(image_array: np.ndarray) -> str:
    """
    Convert numpy image array to base64-encoded PNG bytes.
    
    Args:
        image_array: 2D numpy array of pixel values (H x W), values 0-255
    
    Returns:
        Base64-encoded string of PNG image
    """
    # Ensure uint8
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8) if image_array.max() <= 1 else image_array.astype(np.uint8)
    
    # Convert to PIL Image and encode as PNG
    pil_image = Image.fromarray(image_array, mode='L')
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    
    # Encode to base64
    return base64.b64encode(image_bytes).decode('utf-8')


def base64_to_image(base64_str: str) -> np.ndarray:
    """
    Convert base64-encoded PNG to numpy image array.
    
    Args:
        base64_str: Base64-encoded PNG string
    
    Returns:
        2D numpy array of pixel values (H x W), uint8
    """
    # Decode from base64
    image_bytes = base64.b64decode(base64_str.encode('utf-8'))
    
    # Load from bytes
    buffer = BytesIO(image_bytes)
    pil_image = Image.open(buffer).convert('L')
    
    return np.array(pil_image, dtype=np.uint8)


def resize_image(image_array: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize image array for display.
    
    Args:
        image_array: 2D numpy array (H x W)
        size: Target size as (width, height)
    
    Returns:
        Resized numpy array
    """
    if image_array.dtype != np.uint8:
        u8 = (image_array * 255).clip(0, 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
    else:
        u8 = image_array
    pil_image = Image.fromarray(u8, mode='L')
    resized = pil_image.resize(size, Image.Resampling.NEAREST)
    return np.array(resized, dtype=np.uint8)


def images_equal_pixel(img1: np.ndarray, img2: np.ndarray) -> bool:
    """
    Check if two images are exactly equal (pixel-level comparison).
    
    Args:
        img1: First image as 2D numpy array
        img2: Second image as 2D numpy array
    
    Returns:
        True if images are identical
    """
    return np.array_equal(img1, img2)


def get_image_stats(image_array: np.ndarray) -> dict:
    """
    Get statistics about an image.
    
    Args:
        image_array: 2D numpy array
    
    Returns:
        Dictionary with stats (min, max, mean, std, shape)
    """
    is_float = np.issubdtype(image_array.dtype, np.floating)
    return {
        "height": image_array.shape[0],
        "width": image_array.shape[1],
        "dtype": str(image_array.dtype),
        "min": round(float(image_array.min()), 3) if is_float else int(image_array.min()),
        "max": round(float(image_array.max()), 3) if is_float else int(image_array.max()),
        "mean": float(image_array.mean()),
        "std": float(image_array.std()),
        "size_bytes": image_array.nbytes,
    }
