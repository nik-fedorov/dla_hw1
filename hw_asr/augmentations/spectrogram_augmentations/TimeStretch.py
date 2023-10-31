from torch import Tensor, distributions
import torchaudio

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, min_scale: float = 0.75, max_scale: float = 1.25, n_feat: int = 128):
        self.scale_sampler = distributions.Uniform(low=min_scale, high=max_scale)
        self._aug = torchaudio.transforms.TimeStretch(n_freq=n_feat)

    def __call__(self, data: Tensor):
        return self._aug(data, self.scale_sampler.sample().item())
