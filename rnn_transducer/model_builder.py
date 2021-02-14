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

from rnn_transducer.encoder import Encoder
from rnn_transducer.decoder import Decoder
from rnn_transducer.model import Transducer
import torch


def build_transducer(
        device: torch.device,
        num_vocabs: int,
        input_size: int = 80,
        enc_hidden_size: int = 320,
        dec_hidden_size: int = 512,
        output_size: int = 320,
        num_enc_layers: int = 4,
        num_dec_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        rnn_type: str = 'lstm',
        sos_id: int = 1,
) -> Transducer:
    encoder = build_encoder(
        output_size,
        input_size,
        enc_hidden_size,
        num_enc_layers,
        dropout,
        bidirectional,
        rnn_type,
    )
    decoder = build_decoder(
        device,
        num_vocabs,
        output_size,
        dec_hidden_size,
        num_dec_layers,
        dropout,
        rnn_type,
        sos_id,
    )
    return Transducer(encoder, decoder, num_vocabs, output_size << 1, dec_hidden_size).to(device)


def build_encoder(
        output_size: int = 320,
        input_size: int = 80,
        hidden_size: int = 320,
        num_layers: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
        rnn_type: str = 'lstm',
) -> Encoder:
    return Encoder(
        output_size,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        bidirectional,
        rnn_type,
    )


def build_decoder(
        device: torch.device,
        num_vocabs: int,
        output_size: int = 320,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
        rnn_type: str = 'lstm',
        sos_id: int = 1,
) -> Decoder:
    return Decoder(
        device,
        num_vocabs,
        output_size,
        hidden_size,
        num_layers,
        dropout,
        rnn_type,
        sos_id,
    )
