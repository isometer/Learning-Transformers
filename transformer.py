from shared_imports import *
from positional_encoding import PositionalEncoding
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.reconstruction_layer = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # gonna be completely honest I have no idea what this does.
        # currently not used at all.

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(0).unsqueeze(1)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src):
        # src_mask, tgt_mask = self.generate_mask(src, tgt)
        # for sequence generation tasks, masking will be important
        
        # notice we no longer have tgt as input!
        # src shape should be (seq_len, input_features)

        src = self.input_projection(src)

        src_embedded = self.dropout(self.positional_encoding(src))
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers: 
            # pass the src embedding through each encoder layer in sequence
            enc_output = enc_layer(enc_output)

        output = self.reconstruction_layer(enc_output)
        return output

        