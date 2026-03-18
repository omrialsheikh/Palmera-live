"""
Module: Pose Estimation (DWPose via ONNX Runtime)
Extracts body + hand keypoints from webcam frames.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image


class PoseEstimator:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.config = config
        self.det_session = None
        self.pose_session = None

    def load(self):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        dwpose_dir = self.config.get("models", {}).get("dwpose", "./models/dwpose")

        det_path = os.path.join(dwpose_dir, "yolox_l.onnx")
        pose_path = os.path.join(dwpose_dir, "dw-ll_ucoco_384.onnx")

        self.det_session = ort.InferenceSession(det_path, providers=providers)
        self.pose_session = ort.InferenceSession(pose_path, providers=providers)

        print("[Pose] DWPose ONNX loaded.")

    def _detect_person(self, frame: np.ndarray) -> list:
        """Detect person bounding boxes using YOLOX."""
        h, w = frame.shape[:2]
        input_size = (640, 640)

        # Preprocess
        ratio = min(input_size[0] / h, input_size[1] / w)
        resized = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
        padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized

        blob = padded.astype(np.float32).transpose(2, 0, 1)[None]

        input_name = self.det_session.get_inputs()[0].name
        outputs = self.det_session.run(None, {input_name: blob})

        # Parse detections - return best person bbox
        dets = outputs[0][0]
        if len(dets) == 0:
            return [0, 0, w, h]  # fallback to full frame

        # Find person with highest confidence
        scores = dets[:, 4] * dets[:, 5]
        best_idx = np.argmax(scores)
        bbox = dets[best_idx, :4] / ratio

        return bbox.astype(int).tolist()

    def _estimate_pose(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Estimate keypoints within bounding box."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return np.zeros((133, 3))

        input_size = (384, 288)
        resized = cv2.resize(person_crop, (input_size[1], input_size[0]))
        blob = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32)
        blob = (blob - mean) / std

        input_name = self.pose_session.get_inputs()[0].name
        outputs = self.pose_session.run(None, {input_name: blob})

        return outputs[0]

    def _draw_pose(self, frame: np.ndarray, keypoints: np.ndarray, bbox: list) -> Image.Image:
        """Draw pose skeleton on a blank canvas."""
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Simple visualization - draw keypoint dots
        x1, y1, x2, y2 = bbox
        crop_h, crop_w = y2 - y1, x2 - x1

        if len(keypoints.shape) == 4:
            # Heatmap output - get peak locations
            kps = keypoints[0]
            for i in range(min(kps.shape[0], 133)):
                heatmap = kps[i]
                peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                conf = heatmap[peak]
                if conf > 0.3:
                    px = int(peak[1] / heatmap.shape[1] * crop_w) + x1
                    py = int(peak[0] / heatmap.shape[0] * crop_h) + y1
                    cv2.circle(canvas, (px, py), 3, (0, 255, 0), -1)

        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    def extract(self, frame: np.ndarray) -> dict:
        """
        Extract pose from a BGR frame.
        Returns dict with 'pose_image' and 'keypoints'.
        """
        bbox = self._detect_person(frame)
        keypoints = self._estimate_pose(frame, bbox)
        pose_image = self._draw_pose(frame, keypoints, bbox)

        return {
            "pose_image": pose_image,
            "keypoints": keypoints,
            "bbox": bbox,
            "raw_frame": frame,
        }

    def extract_batch(self, frames: list[np.ndarray]) -> list[dict]:
        return [self.extract(f) for f in frames]
