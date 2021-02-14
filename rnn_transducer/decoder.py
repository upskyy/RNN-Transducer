# Copyright (c) 2021, Sangchun Ha. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import Tensor
from typing import Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn


class Decoder(nn.Module):
    supported_rnns = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU
    }

    def __init__(
            self,
            device: torch.device,
            num_vocabs: int,
            output_size: int = 320,
            hidden_size: int = 512,
            num_layers: int = 1,
            dropout: float = 0.3,
            rnn_type: str = 'lstm',
            sos_id: int = 1,
    ) -> None:
        super(Decoder, self).__init__()
        self.device = device
        self.sos_id = sos_id
        self.embedding = nn.Embedding(num_vocabs, hidden_size)
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(hidden_size, hidden_size, num_layers, True, True, dropout, False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
            self,
            inputs: Tensor,
            inputs_lens: Tensor = None,
            hidden: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        embedded = self.embedding(inputs).to(self.device)

        if inputs_lens is not None:
            embedded = pack_padded_sequence(embedded, inputs_lens, batch_first=True)
            rnn_output, hidden = self.rnn(embedded, hidden)
            rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)

        else:
            rnn_output, hidden = self.rnn(embedded, hidden)

        output = self.fc(rnn_output)

        return output, hidden


