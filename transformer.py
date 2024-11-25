from shared_imports import *
from positional_encoding import PositionalEncoding
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        # self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        # self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # gonna be completely honest I have no idea what this does.

        print("------ DEBUG ---------")
        print(f"tgt has shape {tgt.shape}")
        print(f"unsqueezing... 0")
        tgt_mask = (tgt != 0).unsqueeze(0)
        print(f"mask has shape {tgt_mask.shape}")
        print("------ DEBUG ---------")

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(0).unsqueeze(1)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        # src_mask, tgt_mask = self.generate_mask(src, tgt)
        # for sequence generation tasks, masking will be important
        src_embedded = self.dropout(self.positional_encoding(src))
        tgt_embedded = self.dropout(self.positional_encoding(tgt))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers: 
            # pass the src embedding through each encoder layer in sequence
            enc_output = enc_layer(enc_output)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers: 
            # pass the tgt embedding through each decoder layer in sequence, each time with a copy of the encoder output
            dec_output = dec_layer(dec_output, enc_output)

        output = self.fc(dec_output)
        return output

        