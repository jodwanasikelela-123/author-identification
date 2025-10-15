
import torch
from torch import nn
from models import PYTORCH_LSTM_MODEL_PATH, device, stoi, labels_dict


class AuthorsBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        output_size,
        num_layers=2,
        bidirectional=True,
        dropout=0.5,
        pad_index=0,
        embedding_matrix=None
    ):
        super(AuthorsBiLSTM, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=pad_index
        )

        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
        self.out = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)  # (batch_size, seq_len, embedding_size)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        return self.out(hidden)

print(" ✅ LOADING PYTORCH MODEL!\n") 

INPUT_DIM = len(stoi)
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
OUTPUT_DIM = 1 if len(labels_dict) == 2 else len(labels_dict)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.6321
PAD_IDX = stoi['[pad]']
author_bilstm = AuthorsBiLSTM(
              vocab_size=INPUT_DIM,
              embedding_size=EMBEDDING_DIM,
              hidden_size=HIDDEN_DIM,
              output_size=OUTPUT_DIM,
              num_layers=N_LAYERS,
              bidirectional=BIDIRECTIONAL,
              dropout=DROPOUT,
              pad_index=PAD_IDX
).to(device)
author_bilstm.load_state_dict(torch.load(PYTORCH_LSTM_MODEL_PATH, map_location=device))
print(" ✅ LOADING PYTORCH MODEL DONE!\n")