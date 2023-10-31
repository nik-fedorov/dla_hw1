import torch
from torch import Tensor, distributions
import torchaudio

from hw_asr.augmentations.base import AugmentationBase


class WhiteNoise(AugmentationBase):
    def __init__(self, snr: float):
        self._aug = torchaudio.transforms.AddNoise()
        self.noiser = distributions.Normal(loc=0.0, scale=1.0)
        self.snr = torch.tensor([snr])

    def __call__(self, data: Tensor):
        noise = self.noiser.sample(data.shape)
        return self._aug(data, noise, self.snr)
