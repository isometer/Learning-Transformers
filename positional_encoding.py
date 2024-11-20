from shared_imports import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model) # positional encoding vector
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # indices starting at one
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add the positional encodings to the input tensor
        x = x + self.pe[:, :x.size(1)]
        return x

    # Each row of the "pe" vector represents a position (or time step) within the sequence.
    # Each column of the "pe" vector represents a dimension in the positional encoding space.
    # The values within the "pe" vector are calculated based on the position (time step) and are
    # determined by the chosen encoding function (commonly using sine and cosine functions).

    # TIL that positional encoding for transformers is an absolutely bonkers thing.
    # Whoever invented this was either a genius or insane.