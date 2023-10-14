import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    assert len(dataset_items) > 0

    result_batch = dict()

    result_batch["audio"] = [item["audio"] for item in dataset_items]
    result_batch["duration"] = [item["duration"] for item in dataset_items]
    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    spectrogram_length = [item["spectrogram"].size(2) for item in dataset_items]
    result_batch["spectrogram_length"] = torch.tensor(spectrogram_length)

    text_encoded_length = [item["text_encoded"].size(1) for item in dataset_items]
    result_batch["text_encoded_length"] = torch.tensor(text_encoded_length)

    batch_size = len(dataset_items)
    spec_n_freq = dataset_items[0]["spectrogram"].size(1)
    max_spec_len = max(spectrogram_length)
    spectrogram = torch.full((batch_size, spec_n_freq, max_spec_len), 0.0)
    for i, item in enumerate(dataset_items):
        spectrogram[i, :, :spectrogram_length[i]] = item["spectrogram"]
    result_batch["spectrogram"] = spectrogram

    max_text_len = max(text_encoded_length)
    text_encoded = torch.full((batch_size, max_text_len), 0, dtype=torch.long)
    for i, item in enumerate(dataset_items):
        text_encoded[i, :text_encoded_length[i]] = item["text_encoded"]
    result_batch["text_encoded"] = text_encoded

    return result_batch
