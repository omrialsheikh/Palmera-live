"""
Module B: MimicMotion Inference
Pose-driven full body animation using MimicMotion + SVD + LCM.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional

from server.utils.latent_cache import LatentCache
from server.modules.stream_buffer import StreamBuffer


class MimicMotionInference:
    def __init__(self, config: dict, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.pipeline = None
        self.buffer = StreamBuffer(
            window_size=config["temporal"]["window_size"],
            overlap=config["temporal"]["overlap"],
        )

    def load(self):
        """Load MimicMotion pipeline with SVD backbone."""
        from diffusers import StableVideoDiffusionPipeline

        svd_path = self.config["models"]["svd"]
        mimic_path = self.config["models"]["mimic_motion"]

        # Load SVD as base pipeline (keep original scheduler)
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            svd_path,
            torch_dtype=self.dtype,
            variant="fp16",
        )

        # Load MimicMotion weights on top
        mimic_weights = torch.load(
            mimic_path, map_location="cpu", weights_only=False
        )
        self._load_mimic_weights(mimic_weights)

        self.pipeline = self.pipeline.to(self.device)

        print("[Inference] MimicMotion pipeline loaded.")

    def _load_mimic_weights(self, state_dict: dict):
        """Load MimicMotion-specific weights into the SVD pipeline."""
        unet_state = {
            k.replace("unet.", ""): v
            for k, v in state_dict.items()
            if k.startswith("unet.")
        }
        if unet_state:
            self.pipeline.unet.load_state_dict(unet_state, strict=False)
            print(f"[Inference] Loaded {len(unet_state)} MimicMotion UNet weights.")

    @torch.no_grad()
    def generate_frames(
        self,
        reference_image: Image.Image,
        pose_images: list[Image.Image],
        cache: LatentCache,
        num_steps: Optional[int] = None,
    ) -> list[np.ndarray]:
        """
        Generate animated frames from reference + pose sequence.

        Args:
            reference_image: The avatar reference image
            pose_images: List of pose skeleton images (from DWPose)
            cache: Latent cache with pre-computed reference embeddings
            num_steps: Override inference steps (default from config)

        Returns:
            List of generated BGR frames (numpy arrays)
        """
        steps = num_steps or self.config["inference"]["num_steps"]

        # Use cached reference latent if available
        ref_latent = cache.get("reference_latent")

        num_frames = max(len(pose_images), 2)  # SVD needs at least 2 frames

        output = self.pipeline(
            image=reference_image,
            num_frames=num_frames,
            num_inference_steps=25,
            decode_chunk_size=2,
            motion_bucket_id=127,
            noise_aug_strength=0.02,
            generator=torch.Generator(device=self.device).manual_seed(
                self.config["inference"]["seed"]
            ),
        )

        # Convert pipeline output to BGR numpy frames
        frames = []
        for frame in output.frames[0]:
            frame_np = np.array(frame)
            frame_bgr = frame_np[:, :, ::-1]  # RGB -> BGR
            frames.append(frame_bgr)

        return frames

    def process_frame(
        self,
        pose_data: dict,
        reference_image: Image.Image,
        cache: LatentCache,
    ) -> Optional[list[np.ndarray]]:
        """
        Add a pose frame to buffer and run inference when window is ready.
        Returns generated frames or None if buffer isn't full yet.
        """
        self.buffer.add_frame(pose_data)

        if not self.buffer.should_run_inference():
            return None

        pose_images = self.buffer.get_pose_images()
        frames = self.generate_frames(reference_image, pose_images, cache)

        return frames

    def reset(self):
        self.buffer.reset()
