from torch import nn

from hw_asr.base import BaseModel


# this code is partially adapted from one of DeepSpeech2 implementations:
# https://github.com/SeanNaren/deepspeech.pytorch/blob/master/deepspeech_pytorch/model.py


class SequenceWiseBatchNorm(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.bn(x)
        x = x.view(t, n, -1)
        return x


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional)

    def forward(self, x, output_lengths, h=None):
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
        x, h = self.rnn(x, h)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2)  # (TxNxH*2) -> (TxNxH) by sum
        return x, h


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, num_rnn_layers, rnn_hid_size, rnn_type=nn.LSTM, bidirectional_rnn=True,
                 **batch):
        super().__init__(n_feats, n_class, **batch)

        self.n_class = n_class
        self.num_rnn_layers = num_rnn_layers

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        rnn_input_size = n_feats
        rnn_input_size = (rnn_input_size + 2 * 20 - 41) // 2 + 1
        rnn_input_size = (rnn_input_size + 2 * 10 - 21) // 2 + 1
        rnn_input_size *= 32

        self.rnn_block = nn.Sequential()
        self.rnn_block.append(BatchRNN(rnn_input_size, rnn_hid_size, rnn_type, bidirectional_rnn))
        for _ in range(num_rnn_layers - 1):
            self.rnn_block.append(SequenceWiseBatchNorm(rnn_hid_size))
            self.rnn_block.append(BatchRNN(rnn_hid_size, rnn_hid_size, rnn_type, bidirectional_rnn))

        self.bn = SequenceWiseBatchNorm(rnn_hid_size)
        self.fc = nn.Linear(rnn_hid_size, n_class, bias=False)

    def forward(self, spectrogram, spectrogram_length, hs=None, **batch):
        output_lengths = self.transform_input_lengths(spectrogram_length)
        x = self.conv_block(spectrogram.unsqueeze(1))

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        # if hs is None, create a list of None values corresponding to the number of rnn layers
        if hs is None:
            hs = [None] * self.num_rnn_layers

        new_hs = []
        rnn_index = 0
        for module in self.rnn_block:
            if type(module) == BatchRNN:
                x, h = module(x, output_lengths, hs[rnn_index])
                new_hs.append(h)
                rnn_index += 1
            else:  # SequenceWiseBatchNorm
                x = module(x)

        x = self.bn(x)
        x = self.fc(x)
        x = x.transpose(0, 1)  # NxTxVOCAB
        return {"logits": x, "output_lengths": output_lengths, "new_hs": new_hs}

    def transform_input_lengths(self, input_lengths):
        output_lengths = input_lengths
        for m in self.conv_block.modules():
            if type(m) == nn.modules.conv.Conv2d:
                output_lengths = (output_lengths + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1
        return output_lengths.int()
