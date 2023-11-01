from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size: int, use_lm=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.use_lm = use_lm

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, logits: Tensor, text: List[str], **kwargs):
        cers = []

        if self.use_lm:
            if hasattr(self.text_encoder, 'ctc_beam_search_with_lm'):
                preds = self.text_encoder.ctc_beam_search_with_lm(logits, log_probs_length, self.beam_size)
                for pred_text, target_text in zip(preds, text):
                    cers.append(calc_cer(target_text, pred_text))
            else:
                raise RuntimeError('your text encoder has no attribute ctc_beam_search_with_lm')
        else:
            probs = log_probs.exp().cpu()
            lengths = log_probs_length.detach().numpy()
            for prob_matrix, length, target_text in zip(probs, lengths, text):
                target_text = BaseTextEncoder.normalize_text(target_text)
                if hasattr(self.text_encoder, "ctc_beam_search"):
                    beam_search_hypos = self.text_encoder.ctc_beam_search(prob_matrix, length, self.beam_size)
                    pred_text = beam_search_hypos[0].text   # take the most probable hypo
                else:
                    argmax = torch.argmax(prob_matrix, dim=-1)
                    pred_text = self.text_encoder.decode(argmax[:length])
                cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
