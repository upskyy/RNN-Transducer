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
from rnn_transducer.encoder import Encoder
from rnn_transducer.decoder import Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class JointNet(nn.Module):
    def __init__(
            self,
            num_vocabs: int,
            output_size: int = 640,
            inner_size: int = 512,
    ) -> None:
        super(JointNet, self).__init__()
        self.fc1 = nn.Linear(output_size, inner_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(inner_size, num_vocabs)

    def forward(
            self,
            encoder_output: Tensor,
            decoder_output: Tensor,
    ) -> Tensor:
        if encoder_output.dim() == 3 and decoder_output.dim() == 3:  # Train
            seq_lens = encoder_output.size(1)
            target_lens = decoder_output.size(1)

            encoder_output = encoder_output.unsqueeze(2)
            decoder_output = decoder_output.unsqueeze(1)

            encoder_output = encoder_output.repeat(1, 1, target_lens, 1)
            decoder_output = decoder_output.repeat(1, seq_lens, 1, 1)

        output = torch.cat((encoder_output, decoder_output), dim=-1)

        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)

        return output


class Transducer(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            num_vocabs: int,
            output_size: int = 640,
            inner_size: int = 512,
    ) -> None:
        super(Transducer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.joint = JointNet(num_vocabs, output_size, inner_size)

    def forward(
            self,
            inputs: Tensor,
            input_lens: Tensor,
            targets: Tensor,
            target_lens: Tensor,
    ) -> Tensor:
        encoder_output = self.encoder(inputs, input_lens)
        decoder_output, _ = self.decoder(targets, target_lens)

        output = self.joint(encoder_output, decoder_output)

        return output

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_lens: int) -> Tensor:
        token_list = list()
        hidden = None

        token = torch.LongTensor([[self.decoder.sos_id]])
        if torch.cuda.is_available():
            token = token.cuda()

        for i in range(max_lens):
            decoder_output, hidden = self.decoder(token, hidden=hidden)
            output = self.joint(encoder_output[i].view(-1), decoder_output.view(-1))
            output = F.softmax(output, dim=0)
            prediction_token = output.topk(1)[1]
            token = prediction_token.unsqueeze(1)  # (1, 1)
            prediction_token = int(prediction_token.item())
            token_list.append(prediction_token)

        return torch.LongTensor(token_list)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, inputs_lens: Tensor) -> Tensor:
        outputs = list()

        encoder_outputs = self.encoder(inputs, inputs_lens)
        max_lens = encoder_outputs.size(1)  # torch.stack 하기 위해서

        for encoder_output in encoder_outputs:
            output = self.decode(encoder_output, max_lens)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)

        return outputs  # (B, T)

