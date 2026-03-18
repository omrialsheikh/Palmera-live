"""
Module: Pose Estimation (DWPose)
Extracts 133-point body + hand keypoints from webcam frames.
"""

import cv2
import numpy as np
import torch
from PIL import Image


class PoseEstimator:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.config = config
        self.detector = None
        self.pose_model = None

    def load(self):
        from controlnet_aux import DWposeDetector

        self.detector = DWposeDetector.from_pretrained(
            "yzd-v/DWPose",
            det_filename="yolox_l.onnx",
            pose_filename="dw-ll_ucoco_384.onnx",
        )
        print("[Pose] DWPose loaded.")

    def extract(self, frame: np.ndarray) -> dict:
        """
        Extract pose from a BGR frame.
        Returns dict with 'pose_image' (rendered skeleton) and 'keypoints'.
        """
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        pose_image = self.detector(
            pil_image,
            detect_resolution=512,
            image_resolution=max(pil_image.size),
        )

        return {
            "pose_image": pose_image,
            "raw_frame": frame,
        }

    def extract_batch(self, frames: list[np.ndarray]) -> list[dict]:
        return [self.extract(f) for f in frames]
