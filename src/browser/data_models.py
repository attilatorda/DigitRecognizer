"""Pydantic models for tagged MNIST dataset schema."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class ImageMetadata(BaseModel):
    """Metadata associated with a tagged image."""
    
    source_dataset: str = Field(..., description="Origin: raw_mnist, custom, etc.")
    original_label: Optional[int] = Field(None, description="Original digit label (0-9)")
    original_index: Optional[int] = Field(None, description="Index in source dataset")
    notes: Optional[str] = Field(None, description="User notes for this image")
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_dataset": "raw_mnist",
                "original_label": 7,
                "original_index": 42,
                "notes": "Interesting writing style"
            }
        }


class TaggedImage(BaseModel):
    """Represents a single tagged image in the dataset."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    md5_hash: str = Field(..., description="MD5 hash for duplicate detection")
    pixel_data: str = Field(..., description="Base64-encoded image bytes")
    tags: List[str] = Field(default_factory=list, description="Custom labels/tags for the image")
    metadata: ImageMetadata = Field(..., description="Image metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "md5_hash": "5d41402abc4b2a76b9719d911017c592",
                "pixel_data": "iVBORw0KGgoAAAANSUhEUgAAABw...",
                "tags": ["US-style", "rotated"],
                "metadata": {
                    "source_dataset": "raw_mnist",
                    "original_label": 7,
                    "original_index": 1000,
                    "notes": "Crossed seven"
                }
            }
        }


class DatasetMetadata(BaseModel):
    """Metadata for the entire tagged dataset."""
    
    version: str = Field(default="1.0", description="Format version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    total_images: int = Field(..., description="Number of images in dataset")
    image_height: int = Field(default=28, description="Image height in pixels")
    image_width: int = Field(default=28, description="Image width in pixels")
    duplicate_count: int = Field(default=0, description="Number of duplicate warnings encountered")
    description: Optional[str] = Field(None, description="User description of the dataset")
    
    class Config:
        json_schema_extra = {
            "example": {
                "version": "1.0",
                "created_at": "2026-05-22T12:00:00",
                "total_images": 150,
                "image_height": 28,
                "image_width": 28,
                "duplicate_count": 3,
                "description": "Custom MNIST variant with style labels"
            }
        }


class TaggedDataset(BaseModel):
    """Complete tagged dataset with metadata and all images."""
    
    metadata: DatasetMetadata = Field(..., description="Dataset-level metadata")
    images: List[TaggedImage] = Field(default_factory=list, description="Tagged images")
    
    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "version": "1.0",
                    "created_at": "2026-05-22T12:00:00",
                    "total_images": 2,
                    "image_height": 28,
                    "image_width": 28,
                    "duplicate_count": 0
                },
                "images": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "md5_hash": "5d41402abc4b2a76b9719d911017c592",
                        "pixel_data": "iVBORw0KGgoAAAANSUhEUgAAABw...",
                        "tags": ["US-style"],
                        "metadata": {
                            "source_dataset": "raw_mnist",
                            "original_label": 7,
                            "original_index": 100
                        }
                    }
                ]
            }
        }
    
    def add_image(self, image: TaggedImage) -> None:
        """Add a tagged image to the dataset."""
        self.images.append(image)
        self.metadata.total_images = len(self.images)
    
    def get_image_hashes(self) -> set:
        """Return set of all image hashes in dataset for dedup checking."""
        return {img.md5_hash for img in self.images}
    
    def find_duplicate_hashes(self, new_hash: str) -> Optional[TaggedImage]:
        """Find existing image by hash. Returns TaggedImage if found."""
        for img in self.images:
            if img.md5_hash == new_hash:
                return img
        return None
