"""
Module C: Compositor
Blends the animated avatar onto the user's chosen background
using Poisson blending for natural edges.
"""

import cv2
import numpy as np


class Compositor:
    def __init__(self, config: dict):
        self.config = config
        self.blending = config.get("compositor", {}).get("blending", "poisson")
        self.feather = config.get("compositor", {}).get("edge_feather", 5)
        self.background = None
        self.avatar_mask = None

    def set_background(self, background: np.ndarray):
        """Set the background image (BGR)."""
        self.background = background

    def set_avatar_mask(self, mask: np.ndarray):
        """Set the avatar segmentation mask from Module A."""
        self.avatar_mask = mask

    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply Gaussian feathering to mask edges for smooth blending."""
        if self.feather <= 0:
            return mask

        mask_float = mask.astype(np.float32)
        blurred = cv2.GaussianBlur(mask_float, (0, 0), self.feather)
        return blurred

    def _poisson_blend(
        self, avatar: np.ndarray, background: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Poisson blending (seamless clone) for natural edge integration."""
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Find center of mask for seamless clone
        moments = cv2.moments(mask_uint8)
        if moments["m00"] == 0:
            return avatar

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        center = (cx, cy)

        # Resize background to match avatar
        bg_resized = cv2.resize(background, (avatar.shape[1], avatar.shape[0]))

        result = cv2.seamlessClone(
            avatar, bg_resized, mask_uint8, center, cv2.NORMAL_CLONE
        )

        return result

    def _alpha_blend(
        self, avatar: np.ndarray, background: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Simple alpha blending as fallback."""
        mask_3ch = np.stack([mask] * 3, axis=-1)
        bg_resized = cv2.resize(background, (avatar.shape[1], avatar.shape[0]))
        result = (avatar * mask_3ch + bg_resized * (1 - mask_3ch)).astype(np.uint8)
        return result

    def composite(
        self,
        avatar_frame: np.ndarray,
        mask: np.ndarray = None,
        background: np.ndarray = None,
    ) -> np.ndarray:
        """
        Composite avatar onto background.

        Args:
            avatar_frame: Generated avatar frame (BGR)
            mask: Segmentation mask (if None, uses stored mask)
            background: Background image (if None, uses stored background)

        Returns:
            Composited frame (BGR)
        """
        bg = background if background is not None else self.background
        m = mask if mask is not None else self.avatar_mask

        if bg is None or m is None:
            return avatar_frame

        feathered = self._feather_mask(m.astype(np.float32))

        if self.blending == "poisson":
            return self._poisson_blend(avatar_frame, bg, feathered)
        else:
            return self._alpha_blend(avatar_frame, bg, feathered)
