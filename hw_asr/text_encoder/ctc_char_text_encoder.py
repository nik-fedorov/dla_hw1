from collections import defaultdict
from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.empty_ind = self.char2ind[self.EMPTY_TOK]

    def ctc_decode(self, inds: List[int]) -> str:
        res_tokens = []
        for i in range(len(inds)):
            if inds[i] != self.empty_ind and (i == 0 or inds[i - 1] != inds[i]):
                res_tokens += inds[i]
        return self.decode(res_tokens)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        # assert char_length > 0

        # init starting prob_dict
        prob_dict = {("", self.EMPTY_TOK): 1.0}  # format: (decoded_prefix, last_char): probability

        # recurrently build final prob_dict
        for i in range(len(probs_length)):
            prev_prob_dict = prob_dict
            prob_dict = defaultdict(float)

            for token_ind, token_prob in enumerate(probs[i]):
                token = self.ind2char[token_ind]
                for (prefix, last_char), prob in prev_prob_dict.items():
                    if token == self.EMPTY_TOK:
                        prob_dict[(prefix, token)] += prob * token_prob
                    else:
                        if last_char == token:
                            prob_dict[(prefix, token)] += prob * token_prob
                        else:
                            prob_dict[(prefix + token, token)] += prob * token_prob

                prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:beam_size])

        # build final hypos
        final_hypos_dict = defaultdict(float)
        for (prefix, last_char), prob in prob_dict.items():
            final_hypos_dict[prefix] += prob

        hypos = [Hypothesis(hypo, prob) for hypo, prob in final_hypos_dict.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
