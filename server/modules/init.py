"""
Module A: Initialization
- Segment reference image (SAM)
- Encode reference latents (SVD encoder)
- Store in GPU memory for reuse every frame
"""

import cv2
import numpy as np
import torch
from PIL import Image

from server.utils.latent_cache import LatentCache


class AvatarInit:
    def __init__(self, config: dict, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.sam_predictor = None
        self.image_encoder = None
        self.vae = None
        self.cache = LatentCache(device, dtype)

    def load_sam(self):
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_b"](
            checkpoint=self.config["models"]["sam"]
        )
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)
        print("[Init] SAM loaded.")

    def load_encoders(self):
        from diffusers import AutoencoderKLTemporalDecoder
        from transformers import CLIPVisionModelWithProjection

        svd_path = self.config["models"]["svd"]

        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_path, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)
        self.vae.eval()

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            svd_path, subfolder="image_encoder", torch_dtype=self.dtype
        ).to(self.device)
        self.image_encoder.eval()

        print("[Init] VAE + Image Encoder loaded.")

    def load(self):
        self.load_sam()
        self.load_encoders()

    def segment_avatar(self, image: np.ndarray) -> dict:
        """
        Segment the person from the background using SAM.
        Returns avatar mask and cropped avatar.
        """
        self.sam_predictor.set_image(image)

        h, w = image.shape[:2]
        # Use center point as prompt (assumes person is roughly centered)
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Pick highest confidence mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        return {
            "mask": mask,
            "masked_image": image * mask[:, :, None],
        }

    @torch.no_grad()
    def encode_reference(self, image: Image.Image) -> dict:
        """
        Encode reference image into latent space and CLIP embeddings.
        Stores results in GPU cache for reuse.
        """
        from torchvision import transforms

        # Prepare image for VAE
        transform = transforms.Compose([
            transforms.Resize((self.config["height"], self.config["width"])),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        img_tensor = transform(image).unsqueeze(0).unsqueeze(2)  # B,C,F,H,W
        img_tensor = img_tensor.to(self.device, dtype=self.dtype)

        # Encode to latent space
        latent = self.vae.encode(img_tensor[:, :, 0]).latent_dist.sample()
        self.cache.store("reference_latent", latent)

        # CLIP embedding
        clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        clip_input = clip_transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)
        clip_embeds = self.image_encoder(clip_input).image_embeds
        self.cache.store("clip_embedding", clip_embeds)

        print(f"[Init] Reference encoded. Cache: {self.cache.memory_usage_mb():.1f} MB")

        return {
            "latent": latent,
            "clip_embedding": clip_embeds,
        }

    def process(self, image: np.ndarray) -> dict:
        """
        Full init pipeline: segment + encode reference image.
        """
        segmentation = self.segment_avatar(image)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        embeddings = self.encode_reference(pil_image)

        return {
            "segmentation": segmentation,
            "embeddings": embeddings,
            "cache": self.cache,
        }
