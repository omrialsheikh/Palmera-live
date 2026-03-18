"""
Palmera Live - Main Pipeline Orchestrator
Connects Module A (Init) → Module B (Inference) → Module C (Compositor)
"""

import cv2
import numpy as np
import torch
import time
from PIL import Image
from omegaconf import OmegaConf
from typing import Optional

from server.utils.device import get_device, get_dtype, empty_cache
from server.modules.init import AvatarInit
from server.modules.pose import PoseEstimator
from server.modules.inference import MimicMotionInference
from server.modules.face_enhance import FaceEnhancer
from server.modules.compositor import Compositor


class PalmeraLivePipeline:
    def __init__(self, config_path: str = "configs/pipeline.yaml"):
        self.config = OmegaConf.to_container(
            OmegaConf.load(config_path), resolve=True
        )
        self.device = get_device()
        self.dtype = get_dtype(self.config["dtype"])

        # Modules
        self.init_module = AvatarInit(self.config, self.device, self.dtype)
        self.pose_module = PoseEstimator(self.config, self.device)
        self.inference_module = MimicMotionInference(self.config, self.device, self.dtype)
        self.face_enhancer = FaceEnhancer(self.config, self.device)
        self.compositor = Compositor(self.config)

        # State
        self.reference_image: Optional[Image.Image] = None
        self.is_initialized = False
        self.frame_queue: list[np.ndarray] = []

    def load_models(self):
        """Load all models into GPU memory."""
        print("\n[Pipeline] Loading models...")
        t0 = time.time()

        self.init_module.load()
        self.pose_module.load()
        self.inference_module.load()
        self.face_enhancer.load()

        t1 = time.time()
        print(f"[Pipeline] All models loaded in {t1 - t0:.1f}s")

    def init_avatar(self, image: np.ndarray) -> dict:
        """
        Module A: Initialize avatar from reference image.
        Call once when user uploads a reference image.
        """
        print("[Pipeline] Initializing avatar...")

        result = self.init_module.process(image)
        self.reference_image = Image.fromarray(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

        # Store mask for compositor
        self.compositor.set_avatar_mask(result["segmentation"]["mask"])

        self.is_initialized = True
        print("[Pipeline] Avatar initialized.")

        return {
            "status": "ready",
            "cache_mb": self.init_module.cache.memory_usage_mb(),
        }

    def set_background(self, image: np.ndarray):
        """Set the background image for compositing."""
        self.compositor.set_background(image)

    def process_frame(self, webcam_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single webcam frame through the full pipeline.
        Returns a generated frame or None if buffer isn't ready.

        Flow: Webcam → DWPose → MimicMotion → CodeFormer → Compositor
        """
        if not self.is_initialized:
            return None

        # Step 1: Extract pose from webcam
        pose_data = self.pose_module.extract(webcam_frame)

        # Step 2: Feed to inference module (buffered)
        generated_frames = self.inference_module.process_frame(
            pose_data,
            self.reference_image,
            self.init_module.cache,
        )

        if generated_frames is None:
            return None

        # Step 3 & 4: Enhance face + composite for each generated frame
        output_frames = []
        for frame in generated_frames:
            # Face enhancement (only on face crop)
            enhanced = self.face_enhancer.enhance(frame)

            # Composite onto background
            composited = self.compositor.composite(enhanced)

            output_frames.append(composited)

        # Queue frames for streaming
        self.frame_queue.extend(output_frames)

        return output_frames[-1] if output_frames else None

    def get_next_frame(self) -> Optional[np.ndarray]:
        """Pop the next frame from the output queue."""
        if self.frame_queue:
            return self.frame_queue.pop(0)
        return None

    def has_frames(self) -> bool:
        return len(self.frame_queue) > 0

    def reset(self):
        """Reset pipeline state."""
        self.inference_module.reset()
        self.init_module.cache.clear()
        self.reference_image = None
        self.is_initialized = False
        self.frame_queue.clear()
        empty_cache()
        print("[Pipeline] Reset complete.")
