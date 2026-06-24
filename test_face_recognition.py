import os
import unittest
import numpy as np
import cv2
import torch

from face_detector import HaarFaceDetector, draw_boxes
from siamese_model import EmbeddingNet, TripletLoss, FaceVerifier


class TestHaarFaceDetector(unittest.TestCase):
    def setUp(self):
        self.detector = HaarFaceDetector()

    def test_detector_loads(self):
        self.assertFalse(self.detector.detector.empty())

    def test_detect_on_blank_image_returns_no_faces(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        boxes = self.detector.detect(blank)
        self.assertEqual(len(boxes), 0)

    def test_detect_returns_list_of_tuples(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        boxes = self.detector.detect(blank)
        self.assertIsInstance(boxes, list)

    def test_draw_boxes_does_not_crash(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        boxes = [(10, 10, 50, 50)]
        output = draw_boxes(image, boxes, labels=["Test"])
        self.assertEqual(output.shape, image.shape)


class TestSiameseModel(unittest.TestCase):
    def setUp(self):
        self.model = EmbeddingNet(embedding_dim=64)

    def test_embedding_shape(self):
        x = torch.randn(4, 3, 128, 128)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 64))

    def test_embeddings_are_l2_normalized(self):
        x = torch.randn(2, 3, 128, 128)
        out = self.model(x)
        norms = out.norm(p=2, dim=1)
        for n in norms:
            self.assertAlmostEqual(n.item(), 1.0, places=4)

    def test_triplet_loss_decreases_for_better_embeddings(self):
        criterion = TripletLoss(margin=0.3)
        anchor = torch.tensor([[1.0, 0.0]])
        close_positive = torch.tensor([[0.95, 0.05]])  # near anchor -> small pos_dist
        close_negative = torch.tensor([[0.9, 0.1]])     # negative also near anchor -> bad: small neg_dist
        far_negative = torch.tensor([[-1.0, 0.0]])       # negative far from anchor -> good: large neg_dist

        good_loss = criterion(anchor, close_positive, far_negative)
        bad_loss = criterion(anchor, close_positive, close_negative)
        self.assertLess(good_loss.item(), bad_loss.item())

    def test_face_verifier_same_embedding_is_match(self):
        verifier = FaceVerifier(self.model, threshold=0.8)
        face = torch.randn(3, 128, 128)
        result = verifier.verify(face, face)  # comparing identical input to itself
        self.assertEqual(result["distance"], 0.0)
        self.assertTrue(result["same_person"])


if __name__ == "__main__":
    unittest.main()
