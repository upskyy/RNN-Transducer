from rnn_transducer.model_builder import build_transducer
import torch
import warnings

warnings.filterwarnings('ignore')

batch_size = 4
seq_length = 1200
input_size = 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.FloatTensor(batch_size, input_size, seq_length).to(device)
input_lengths = torch.LongTensor([seq_length, seq_length - 10, seq_length - 20, seq_length - 30]).to(device)

model = build_transducer(
        device,
        num_vocabs=10,
        input_size=input_size,
)

outputs = model.recognize(inputs, input_lengths)
print(outputs)
# tensor([[0, 0, 0,  ..., 9, 9, 9],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0]])
print(outputs.size())  # torch.Size([4, 1200])
