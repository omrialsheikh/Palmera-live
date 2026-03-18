"""
Module B: MimicMotion Inference
Uses the actual MimicMotion pipeline (not vanilla SVD).
MimicMotion = SVD + PoseNet + confidence-aware pose conditioning.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import Optional

from server.utils.latent_cache import LatentCache
from server.modules.stream_buffer import StreamBuffer

# Add MimicMotion repo to path
MIMIC_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "MimicMotion")
if os.path.exists(MIMIC_REPO):
    sys.path.insert(0, MIMIC_REPO)


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
        """Load MimicMotion pipeline with PoseNet + modified UNet."""
        from mimicmotion.utils.loader import create_pipeline
        from omegaconf import OmegaConf

        # Build config that MimicMotion expects
        infer_config = OmegaConf.create({
            "base_model_path": self.config["models"]["svd"],
            "ckpt_path": self.config["models"]["mimic_motion"],
        })

        self.pipeline = create_pipeline(infer_config, self.device)

        # Convert entire pipeline to fp16 to match checkpoint weights
        self.pipeline = self.pipeline.to(device=self.device, dtype=self.dtype)

        print("[Inference] MimicMotion pipeline loaded (with PoseNet).")

    @torch.no_grad()
    def generate_frames(
        self,
        reference_image: Image.Image,
        pose_images: list[Image.Image],
        cache: LatentCache,
        num_steps: Optional[int] = None,
    ) -> list[np.ndarray]:
        """
        Generate animated frames using MimicMotion pipeline.

        Args:
            reference_image: The avatar reference image (PIL)
            pose_images: List of pose skeleton images from DWPose (PIL)
            cache: Latent cache (for future optimization)
            num_steps: Override inference steps

        Returns:
            List of generated BGR frames (numpy arrays)
        """
        steps = num_steps or self.config["inference"]["num_steps"]
        num_frames = len(pose_images)

        # Convert pose images to tensor [F, H, W, 3] normalized to [-1, 1]
        h, w = self.config["height"], self.config["width"]

        pose_pixels = []
        for pose_img in pose_images:
            pose_resized = pose_img.resize((w, h))
            pose_np = np.array(pose_resized).astype(np.float32) / 127.5 - 1.0
            pose_pixels.append(pose_np)
        pose_tensor = torch.tensor(np.stack(pose_pixels)).to(self.device, dtype=self.dtype)

        # Resize reference image
        ref_resized = reference_image.resize((w, h))

        # Run MimicMotion pipeline
        output = self.pipeline(
            ref_resized,
            image_pose=pose_tensor,
            num_frames=num_frames,
            tile_size=min(num_frames, 16),
            tile_overlap=min(num_frames // 2, 6),
            height=h,
            width=w,
            fps=15,
            noise_aug_strength=0.0,
            num_inference_steps=steps,
            generator=torch.Generator(device=self.device).manual_seed(
                self.config["inference"]["seed"]
            ),
            min_guidance_scale=2.0,
            max_guidance_scale=2.0,
            decode_chunk_size=2,
            output_type="pt",
            device=self.device,
        )

        # Convert output frames to BGR numpy
        frames = []
        video_tensor = output.frames  # [B, F, H, W, 3] uint8
        if isinstance(video_tensor, torch.Tensor):
            video_np = video_tensor[0].cpu().numpy()
        else:
            video_np = np.array(video_tensor[0])

        for i in range(video_np.shape[0]):
            frame_rgb = video_np[i]
            if frame_rgb.max() <= 1.0:
                frame_rgb = (frame_rgb * 255).astype(np.uint8)
            frame_bgr = frame_rgb[:, :, ::-1]  # RGB -> BGR
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
