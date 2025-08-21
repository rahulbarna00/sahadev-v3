import torch
import torch.nn.functional as F
import numpy as np
import cv2
from filters import get_directional_filters


class ImageDehazer:
    def __init__(self, lambda_reg=0.1, sigma=0.5, delta=0.85):
        self.lambda_reg = lambda_reg
        self.sigma = sigma
        self.delta = delta
        self.filters = get_directional_filters()  # [N, 1, 3, 3]

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.squeeze(0).permute(1, 2, 0)
        return (tensor.clamp(0, 1).numpy() * 255).astype(np.uint8)

    def estimate_transmission(self, img: torch.Tensor) -> torch.Tensor:
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        dark_channel = torch.min(torch.stack([r, g, b], dim=1), dim=1)[0]
        t = 1 - self.delta * dark_channel
        return t.unsqueeze(1)

    def refine_transmission(self, t: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        t_ref = t.clone()
        I = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]  # grayscale
        I = I.unsqueeze(1)  # [B, 1, H, W]

        for fk in self.filters:
            fk = fk.to(t.device)
            fk = fk.contiguous()  # make sure filter is memory contiguous
            print(f"fk shape: {fk.shape}, t_ref: {t_ref.shape}, I: {I.shape}")
            dt = F.conv2d(t_ref, fk, padding=1, stride=1)
            di = F.conv2d(I, fk, padding=1, stride=1)
            wt = torch.exp(- (di ** 2) / (2 * self.sigma ** 2))
            t_ref = t_ref + self.lambda_reg * F.conv2d(wt * dt, -fk, padding=1, stride=1)

        return torch.clamp(t_ref, 0, 1)

    def recover_radiance(self, img: torch.Tensor, t: torch.Tensor, A=1.0) -> torch.Tensor:
        t = t.clamp(min=0.01)
        J = (img - A * (1 - t)) / t
        return torch.clamp(J, 0, 1)

    def dehaze(self, img_np: np.ndarray) -> tuple:
        img = self.to_tensor(img_np)
        t_est = self.estimate_transmission(img)
        t_ref = self.refine_transmission(t_est, img)
        J = self.recover_radiance(img, t_ref)
        return self.to_numpy(J), self.to_numpy(t_ref.repeat(1, 3, 1, 1))
