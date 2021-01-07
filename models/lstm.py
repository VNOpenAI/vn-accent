import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
            d_model, bidirectional=False):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, d_model)
        num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(d_model, d_model, bidirectional=bidirectional, batch_first=True)
        self.out = nn.Linear(d_model * num_directions, trg_vocab_size)
    def forward(self, src):
        embedded = self.embed(src)
        lstm_output, hidden = self.lstm(embedded)
        output = self.out(lstm_output)
        return output
