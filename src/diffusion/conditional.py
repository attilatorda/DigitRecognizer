"""Class-conditional wrapper around denoising-diffusion-pytorch's GaussianDiffusion.

The library has no built-in class conditioning. This module injects a learned
nn.Embedding into the Unet's time-step signal by permanently replacing the
unet.time_mlp attribute with a CondTimeMlp wrapper at construction time.
"""

import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


class CondTimeMlp(nn.Module):
    """
    Drop-in replacement for Unet.time_mlp that adds a class embedding to
    the time embedding. The class embedding tensor is set externally before
    each forward/sample call via _current_class_emb.
    """

    def __init__(self, original_mlp: nn.Module):
        super().__init__()
        self.original = original_mlp
        self._current_class_emb: torch.Tensor | None = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out = self.original(t)
        if self._current_class_emb is not None:
            out = out + self._current_class_emb
        return out


class ConditionalGaussianDiffusion(nn.Module):
    """
    Class-conditional DDPM built on top of denoising-diffusion-pytorch.

    Usage:
        unet  = Unet(dim=32, channels=1, dim_mults=(1,2,4))
        model = ConditionalGaussianDiffusion(
                    unet, num_classes=17, image_size=28,
                    timesteps=1000, sampling_timesteps=100)

        # Training
        loss = model(images, classes)        # images (B,1,28,28), classes (B,)
        loss.backward()

        # Sampling (DDIM)
        samples = model.sample(classes)      # returns (B,1,28,28) tensor
    """

    def __init__(
        self,
        unet: Unet,
        num_classes: int,
        image_size: int = 28,
        timesteps: int = 1000,
        sampling_timesteps: int = 100,
        **diffusion_kwargs,
    ):
        super().__init__()

        time_dim: int = unet.time_mlp[-1].out_features
        self.class_emb = nn.Embedding(num_classes, time_dim)

        # Permanently replace unet.time_mlp with the conditioned wrapper
        self.cond_time_mlp = CondTimeMlp(unet.time_mlp)
        unet.time_mlp = self.cond_time_mlp

        self.diffusion = GaussianDiffusion(
            unet,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            **diffusion_kwargs,
        )

    def forward(self, img: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """Compute the diffusion training loss for a batch of images and class labels."""
        self.cond_time_mlp._current_class_emb = self.class_emb(classes)
        try:
            loss = self.diffusion(img)
        finally:
            self.cond_time_mlp._current_class_emb = None
        return loss

    @torch.no_grad()
    def sample(self, classes: torch.Tensor) -> torch.Tensor:
        """Generate one image per class label using DDIM sampling."""
        self.cond_time_mlp._current_class_emb = self.class_emb(classes)
        try:
            samples = self.diffusion.sample(batch_size=len(classes))
        finally:
            self.cond_time_mlp._current_class_emb = None
        return samples

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device | None = None) -> None:
        self.load_state_dict(torch.load(path, map_location=device or "cpu"))
