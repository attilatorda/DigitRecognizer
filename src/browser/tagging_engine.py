"""Core tagging engine for MNIST browser session management."""

import numpy as np
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from .data_models import TaggedImage, TaggedDataset, DatasetMetadata, ImageMetadata
from .image_utils import hash_image, image_to_base64, base64_to_image


class TaggingSession:
    """
    Manages a tagging session: loads images, manages tags, detects duplicates, exports/imports.
    """
    
    def __init__(self, images: np.ndarray, labels: Optional[np.ndarray] = None, 
                 source_dataset: str = "raw_mnist"):
        """
        Initialize a tagging session.
        
        Args:
            images: Array of images (N x H x W) - numpy uint8 array
            labels: Optional array of original labels (N,)
            source_dataset: Name of source dataset (raw_mnist, variants14, etc.)
        """
        self.images = images
        self.labels = labels
        self.source_dataset = source_dataset
        self.num_images = len(images)
        
        # Current session state
        self.tagged_dataset = TaggedDataset(
            metadata=DatasetMetadata(
                total_images=0,
                description=f"Tagged subset from {source_dataset}"
            )
        )
        self.current_tags: Dict[int, set] = {}  # image_index -> set of tags
        self.duplicate_warnings: List[Dict[str, Any]] = []
        self.duplicate_hashes: Dict[str, int] = {}  # hash -> image_index (for session dedup)
        self._source_index_to_tagged_pos: Dict[int, int] = {}
    
    def add_image(self, image_index: int, tags: Optional[List[str]] = None,
                  existing_dataset: Optional[TaggedDataset] = None,
                  user_notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Add an image to the tagged dataset with tags.
        
        Args:
            image_index: Index of image in self.images
            tags: List of tags to add. If None, uses empty list.
            existing_dataset: Optional existing dataset to check for duplicates
            user_notes: Optional user notes for this image
        
        Returns:
            Dictionary with result: {
                'success': bool,
                'duplicate_found': bool,
                'duplicate_info': optional dict if duplicate found,
                'image_id': str,
                'message': str
            }
        """
        if image_index < 0 or image_index >= self.num_images:
            return {
                'success': False,
                'duplicate_found': False,
                'message': f"Invalid image index: {image_index}"
            }
        
        image = self.images[image_index]
        image_hash = hash_image(image)

        # If image is already in session, merge tags and update notes instead of adding duplicate row
        existing_pos = self._source_index_to_tagged_pos.get(image_index)
        if existing_pos is not None:
            existing_img = self.tagged_dataset.images[existing_pos]
            incoming_tags = [t for t in (tags or []) if t]
            merged_tags = list(dict.fromkeys(existing_img.tags + incoming_tags))
            existing_img.tags = merged_tags
            self.current_tags[image_index] = set(merged_tags)

            if user_notes:
                existing_img.metadata.notes = user_notes

            return {
                'success': True,
                'duplicate_found': False,
                'duplicate_info': None,
                'image_id': existing_img.id,
                'message': f"Image already in session; merged tags ({len(incoming_tags)} incoming)"
            }
        
        # Check for duplicate in existing dataset
        duplicate_info = None
        if existing_dataset:
            dup_img = existing_dataset.find_duplicate_hashes(image_hash)
            if dup_img:
                duplicate_info = {
                    'duplicate_id': dup_img.id,
                    'duplicate_tags': dup_img.tags,
                    'duplicate_index': image_index,
                    'message': f"Image already exists in dataset with ID {dup_img.id}"
                }
                self.duplicate_warnings.append(duplicate_info)
        
        # Create tagged image
        tags = tags or []
        tagged_image = TaggedImage(
            md5_hash=image_hash,
            pixel_data=image_to_base64(image),
            tags=tags,
            metadata=ImageMetadata(
                source_dataset=self.source_dataset,
                original_label=int(self.labels[image_index]) if self.labels is not None else None,
                original_index=image_index,
                notes=user_notes
            )
        )
        
        self.tagged_dataset.add_image(tagged_image)
        self.current_tags[image_index] = set(tags)
        self._source_index_to_tagged_pos[image_index] = len(self.tagged_dataset.images) - 1
        if image_hash not in self.duplicate_hashes:
            self.duplicate_hashes[image_hash] = image_index
        
        return {
            'success': True,
            'duplicate_found': duplicate_info is not None,
            'duplicate_info': duplicate_info,
            'image_id': tagged_image.id,
            'message': f"Image added successfully with {len(tags)} tag(s)"
        }
    
    def add_tag(self, image_index: int, tag: str) -> bool:
        """
        Add a tag to an image already in the session.
        
        Args:
            image_index: Index of image
            tag: Tag string to add
        
        Returns:
            True if successful, False if image not in session
        """
        if image_index not in self.current_tags:
            return False

        pos = self._source_index_to_tagged_pos.get(image_index)
        if pos is None or pos >= len(self.tagged_dataset.images):
            return False

        cleaned_tag = tag.strip()
        if not cleaned_tag:
            return False

        if cleaned_tag not in self.tagged_dataset.images[pos].tags:
            self.tagged_dataset.images[pos].tags.append(cleaned_tag)
            self.current_tags[image_index].add(cleaned_tag)
        return True
    
    def remove_tag(self, image_index: int, tag: str) -> bool:
        """
        Remove a tag from an image in the session.
        
        Args:
            image_index: Index of image (position in tagged_dataset.images)
            tag: Tag string to remove
        
        Returns:
            True if successful, False if tag not found
        """
        pos = self._source_index_to_tagged_pos.get(image_index)
        if pos is not None and pos < len(self.tagged_dataset.images):
            img = self.tagged_dataset.images[pos]
            if tag in img.tags:
                img.tags.remove(tag)
                if image_index in self.current_tags:
                    self.current_tags[image_index].discard(tag)
                return True
        return False
    
    def get_duplicate_candidates(self, image_index: int, 
                                existing_dataset: Optional[TaggedDataset] = None) -> List[Dict[str, Any]]:
        """
        Find potential duplicate images (by hash).
        
        Args:
            image_index: Index of image to check
            existing_dataset: Optional existing dataset to cross-check
        
        Returns:
            List of candidate duplicates with their info
        """
        if image_index >= len(self.images):
            return []
        
        image = self.images[image_index]
        image_hash = hash_image(image)
        candidates = []
        
        # Check in existing dataset
        if existing_dataset:
            dup_img = existing_dataset.find_duplicate_hashes(image_hash)
            if dup_img:
                candidates.append({
                    'location': 'existing_dataset',
                    'id': dup_img.id,
                    'tags': dup_img.tags,
                    'metadata': dup_img.metadata.dict()
                })
        
        # Check in current session
        if image_hash in self.duplicate_hashes:
            session_idx = self.duplicate_hashes[image_hash]
            if session_idx != image_index:
                candidates.append({
                    'location': 'current_session',
                    'session_index': session_idx,
                    'tags': list(self.current_tags.get(session_idx, set()))
                })
        
        return candidates
    
    def export_to_json(self, output_path: str) -> Dict[str, Any]:
        """
        Export tagged dataset to JSON file.
        
        Args:
            output_path: Path to output JSON file
        
        Returns:
            Result dictionary with success status and message
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize dataset with proper datetime handling
            dataset_dict = self.tagged_dataset.model_dump(mode='json')
            
            with open(output_file, 'w') as f:
                json.dump(dataset_dict, f, indent=2)
            
            return {
                'success': True,
                'message': f"Exported {len(self.tagged_dataset.images)} images to {output_path}",
                'file_size': output_file.stat().st_size,
                'num_images': len(self.tagged_dataset.images)
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Export failed: {str(e)}"
            }
    
    @staticmethod
    def load_from_json(json_path: str) -> tuple[Optional['TaggingSession'], str]:
        """
        Load a previously saved tagged dataset from JSON.
        
        Args:
            json_path: Path to JSON file
        
        Returns:
            Tuple of (TaggingSession or None, message string)
        """
        try:
            json_file = Path(json_path)
            if not json_file.exists():
                return None, f"File not found: {json_path}"
            
            with open(json_file, 'r') as f:
                dataset_dict = json.load(f)
            
            # Deserialize dataset
            dataset = TaggedDataset(**dataset_dict)
            
            # Create session with dummy images (loaded images aren't stored in JSON)
            session = TaggingSession(
                images=np.zeros((len(dataset.images), 28, 28), dtype=np.uint8),
                labels=None,
                source_dataset="loaded_from_json"
            )
            session.tagged_dataset = dataset
            
            # Rebuild hashes and source-index mapping from loaded images
            for idx, img in enumerate(dataset.images):
                source_idx = img.metadata.original_index if img.metadata.original_index is not None else idx
                if img.md5_hash not in session.duplicate_hashes:
                    session.duplicate_hashes[img.md5_hash] = source_idx
                session.current_tags[source_idx] = set(img.tags)
                session._source_index_to_tagged_pos[source_idx] = idx
            
            return session, (
                f"Loaded {len(dataset.images)} images from {json_path}. "
                "Note: image arrays are placeholders until a real source dataset is loaded."
            )
        except Exception as e:
            return None, f"Load failed: {str(e)}"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current tagging session."""
        unique_tags = set()
        for tags in self.current_tags.values():
            unique_tags.update(tags)
        
        return {
            'total_images_available': self.num_images,
            'images_tagged': len(self.tagged_dataset.images),
            'unique_tags': list(unique_tags),
            'num_unique_tags': len(unique_tags),
            'avg_tags_per_image': (
                sum(len(tags) for tags in self.current_tags.values()) / len(self.current_tags)
                if self.current_tags else 0
            ),
            'duplicate_warnings': len(self.duplicate_warnings),
            'source_dataset': self.source_dataset,
        }
