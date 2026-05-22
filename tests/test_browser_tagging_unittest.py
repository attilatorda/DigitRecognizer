"""Automated unit tests for browser tagging engine behavior."""

import unittest
import numpy as np

from src.browser import TaggingSession


class TestBrowserTaggingEngine(unittest.TestCase):
    def setUp(self):
        self.images = np.random.randint(0, 256, size=(200, 28, 28), dtype=np.uint8)
        self.labels = np.random.randint(0, 10, size=(200,), dtype=np.uint8)

    def test_readd_same_index_merges_tags_without_duplicate_rows(self):
        session = TaggingSession(self.images, self.labels, source_dataset="unit_test")

        r1 = session.add_image(5, tags=["US-style", "rotated"])
        self.assertTrue(r1["success"])
        self.assertEqual(len(session.tagged_dataset.images), 1)

        r2 = session.add_image(5, tags=["crossed", "US-style"])
        self.assertTrue(r2["success"])
        self.assertEqual(len(session.tagged_dataset.images), 1)

        tagged = session.tagged_dataset.images[0]
        self.assertIn("US-style", tagged.tags)
        self.assertIn("rotated", tagged.tags)
        self.assertIn("crossed", tagged.tags)
        self.assertEqual(tagged.tags.count("US-style"), 1)

    def test_add_remove_tag_works_for_sparse_source_indices(self):
        session = TaggingSession(self.images, self.labels, source_dataset="unit_test")

        r = session.add_image(100, tags=["baseline"])
        self.assertTrue(r["success"])

        self.assertTrue(session.add_tag(100, "italic"))
        tagged = session.tagged_dataset.images[0]
        self.assertIn("italic", tagged.tags)

        self.assertTrue(session.remove_tag(100, "baseline"))
        self.assertNotIn("baseline", tagged.tags)

    def test_duplicate_candidate_detects_current_session_hash_match(self):
        # Force image 20 to be identical to image 10
        images = self.images.copy()
        images[20] = images[10].copy()

        session = TaggingSession(images, self.labels, source_dataset="unit_test")
        r = session.add_image(10, tags=["canonical"])
        self.assertTrue(r["success"])

        candidates = session.get_duplicate_candidates(20)
        self.assertTrue(any(c["location"] == "current_session" for c in candidates))


if __name__ == "__main__":
    unittest.main()
