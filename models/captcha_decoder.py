import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptchaDecoder(nn.Module):
    def __init__(self, hidden_size=2048, n_classes=26, sequence_len=5):
        super(CaptchaDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len
        self.embedding = nn.Embedding(n_classes, hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, state, device):  # n x hidden_size
        batch_size = state.size(0)
        inputs = torch.LongTensor([0] * batch_size).unsqueeze(0).to(device)  # 1 x n
        outputs = []
        all_symbols = []
        for _ in range(self.sequence_len):
            embedded = self.embedding(inputs)  # 1 x n x hidden_size
            output, state = self.rnn(embedded, state.unsqueeze(0))  # 1 x n x hidden_size
            predicted_softmax = F.log_softmax(self.linear(output.view(-1, self.hidden_size)), dim=1)  # n x classes
            state = state.squeeze(0)
            outputs.append(predicted_softmax)
            symbols = predicted_softmax.topk(1)[1]  # n x 1
            inputs = symbols.view(1, -1)
            all_symbols.append(symbols.view(-1))
        return torch.stack(outputs), torch.stack(all_symbols)  # seq x n x classes, seq x n
