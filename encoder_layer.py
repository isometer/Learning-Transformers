from shared_imports import *
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) # dropout helps ensure model robustness

    def forward(self, x, mask):
        # step 1: self-attention
        attn_output = self.self_attn(x, x, x, mask)
        # step 2: residual connection and layer normalization
        x - self.norm1(x + self.dropout(attn_output))
        # step 3: position-wise feedforward network
        ff_output = self.feed_forward(x)
        # step 4: residual connection and layer normalization
        x = self.norm2(x + self.dropout(ff_output))
        return x