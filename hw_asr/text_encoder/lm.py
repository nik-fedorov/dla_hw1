import logging
import gzip
import multiprocessing
import os
import shutil
from pathlib import Path

import pyctcdecode
import wget

VOCAB_URL = 'https://www.openslr.org/resources/11/librispeech-vocab.txt'
LM_URL = 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'

VOCAB_FILENAME = 'librispeech-vocab.txt'
ZIPPED_FILENAME = '3-gram.pruned.1e-7.arpa.gz'
UNZIPPED_FILENAME = '3-gram.pruned.1e-7.arpa'
LOWERCASE_LM_FILENAME = 'lowercase_3-gram.pruned.1e-7.arpa'

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent
LM_DIR = ROOT_PATH / 'hw_asr' / 'text_encoder' / 'lm'


logger = logging.getLogger(__name__)


# this code is based on tutorial
# https://github.com/kensho-technologies/pyctcdecode/blob/main/tutorials/03_eval_performance.ipynb


class LM:
    def __init__(self, alphabet, lm_dir=LM_DIR):
        self.lm_dir = Path(lm_dir)
        self._load_lm_and_vocab()
        self.decoder = pyctcdecode.build_ctcdecoder(
            alphabet,
            unigrams=self._get_unigrams(self.lm_dir / VOCAB_FILENAME),
            kenlm_model_path=str(self.lm_dir / LOWERCASE_LM_FILENAME)
        )

    def decode(self, logits, probs_length, beam_size):
        batch_probs = [logits_sample[:length].cpu().numpy() for logits_sample, length in zip(logits, probs_length)]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = self.decoder.decode_batch(pool, batch_probs, beam_width=beam_size)
        return results

    def _load_lm_and_vocab(self):
        # ensure lm_dir exists
        self.lm_dir.mkdir(exist_ok=True)

        # load zipped lm
        if not os.path.exists(self.lm_dir / ZIPPED_FILENAME):
            logger.info('Downloading pruned 3-gram model.')
            wget.download(LM_URL, out=str(self.lm_dir))
            logger.info('Downloaded the 3-gram language model.')
        else:
            logger.info('Pruned .arpa.gz already exists.')

        # unzip
        if not os.path.exists(self.lm_dir / UNZIPPED_FILENAME):
            with gzip.open(self.lm_dir / ZIPPED_FILENAME, 'rb') as f_zipped:
                with open(self.lm_dir / UNZIPPED_FILENAME, 'wb') as f_unzipped:
                    shutil.copyfileobj(f_zipped, f_unzipped)
            logger.info('Unzipped the 3-gram language model.')
        else:
            logger.info('Unzipped .arpa already exists.')

        # lowercase lm
        if not os.path.exists(self.lm_dir / LOWERCASE_LM_FILENAME):
            with open(self.lm_dir / UNZIPPED_FILENAME, 'r') as f_upper:
                with open(self.lm_dir / LOWERCASE_LM_FILENAME, 'w') as f_lower:
                    for line in f_upper:
                        f_lower.write(line.lower())

        # load vocab
        if not os.path.exists(self.lm_dir / VOCAB_FILENAME):
            wget.download(VOCAB_URL, out=str(self.lm_dir))
            logger.info('Vocab downloaded.')
        else:
            logger.info('Vocab is already downloaded.')

    def _get_unigrams(self, vocab_path):
        with open(vocab_path) as f:
            unigram_list = [t.lower() for t in f.read().strip().split("\n")]
        return unigram_list
