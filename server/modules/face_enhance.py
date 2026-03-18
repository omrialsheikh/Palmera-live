"""
Face Enhancement using CodeFormer.
Runs only on the face-crop region to fix eyes/teeth artifacts.
"""

import cv2
import numpy as np
import torch


class FaceEnhancer:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.enabled = config.get("face_enhance", {}).get("enabled", True)
        self.fidelity = config.get("face_enhance", {}).get("fidelity", 0.7)
        self.padding = config.get("face_enhance", {}).get("crop_padding", 1.3)
        self.restorer = None

    def load(self):
        if not self.enabled:
            print("[FaceEnhance] Disabled in config.")
            return

        from basicsr.utils import img2tensor, tensor2img
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            device=self.device,
        )

        from basicsr.archs.codeformer_arch import CodeFormer

        self.net = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)

        ckpt = torch.load(
            self.config["models"]["codeformer"],
            map_location="cpu",
        )
        self.net.load_state_dict(ckpt["params_ema"])
        self.net.eval()

        print("[FaceEnhance] CodeFormer loaded.")

    @torch.no_grad()
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance face region in a BGR frame.
        Returns the frame with enhanced face pasted back.
        """
        if not self.enabled or self.net is None:
            return frame

        from basicsr.utils import img2tensor, tensor2img

        self.face_helper.clean_all()
        self.face_helper.read_image(frame)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()

        if len(self.face_helper.cropped_faces) == 0:
            return frame

        for cropped_face in self.face_helper.cropped_faces:
            face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            face_t = face_t.unsqueeze(0).to(self.device)

            output = self.net(face_t, w=self.fidelity, adain=True)[0]
            restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(0, 1))
            restored_face = restored_face.astype("uint8")

            self.face_helper.add_restored_face(restored_face)

        self.face_helper.get_inverse_affine(None)
        result = self.face_helper.paste_faces_to_image()

        return result
