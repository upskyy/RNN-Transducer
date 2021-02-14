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
import torch.nn as nn


class Encoder(nn.Module):
    supported_rnns = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU
    }

    def __init__(
            self,
            output_size: int = 320,
            input_size: int = 80,
            hidden_size: int = 320,
            num_layers: int = 4,
            dropout: float = 0.3,
            bidirectional: bool = True,
            rnn_type: str = 'lstm',
    ) -> None:
        super(Encoder, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_size, num_layers, True, True, dropout, bidirectional)
        self.rnn_output_size = hidden_size << 1 if bidirectional else hidden_size
        self.fc = nn.Linear(self.rnn_output_size, output_size)

    def forward(
            self,
            inputs: Tensor,
            inputs_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        inputs = inputs.transpose(1, 2)

        inputs = pack_padded_sequence(inputs, inputs_lens, batch_first=True)
        rnn_output, _ = self.rnn(inputs)
        rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)

        output = self.fc(rnn_output)

        return output, inputs_lens
