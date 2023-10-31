from typing import List, Callable

import hw_asr.augmentations.spectrogram_augmentations
import hw_asr.augmentations.wave_augmentations
from hw_asr.augmentations.random_apply import RandomApply
from hw_asr.augmentations.sequential import SequentialAugmentation
from hw_asr.utils.parse_config import ConfigParser


def from_configs(configs: ConfigParser):
    wave_augs = []
    if "augmentations" in configs.config and "wave" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["wave"]:
            wave_augs.append(
                configs.init_obj(aug_dict, hw_asr.augmentations.wave_augmentations)
            )

    spec_augs = []
    if "augmentations" in configs.config and "spectrogram" in configs.config["augmentations"]:
        for aug_dict in configs.config["augmentations"]["spectrogram"]:
            spec_augs.append(
                configs.init_obj(aug_dict, hw_asr.augmentations.spectrogram_augmentations)
            )

    probability = 1.0
    if "augmentations" in configs.config and "p" in configs.config["augmentations"]:
        probability = configs.config["augmentations"]["probability"]

    return _to_function(wave_augs, probability), _to_function(spec_augs, probability)


def _to_function(augs_list: List[Callable], prob: int):
    if len(augs_list) == 0:
        return None
    elif len(augs_list) == 1:
        return RandomApply(augs_list[0], prob)
    else:
        augs_list = [RandomApply(aug, prob) for aug in augs_list]
        return SequentialAugmentation(augs_list)
