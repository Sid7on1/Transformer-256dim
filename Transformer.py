import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer

# 1. Tokenizer and Input
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
text = "hi cat are you okay"
inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=12)
input_ids = inputs['input_ids']  # [1, 12]

# 2. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# 3. Embedding Layer
embedding_dim = 256
vocab_size = tokenizer.vocab_size
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# 4. Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [batch, heads, seq, seq]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)  # [batch, heads, seq, head_dim]
    return output, attn

# 5. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()
        
        Q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_linear(attn_output)
        return output, attn_weights

# 6. Feed Forward Network
class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# 7. Transformer Encoder Layer with Add & Norm
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super().__init__()
        self.mha = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = FeedForwardNetwork(embedding_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, mask=None):
        # Multi-head attention + Add & Norm
        attn_output, _ = self.mha(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed Forward + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

# === Putting it all together ===

# Get embeddings
embeddings = embedding_layer(input_ids)  # [1, 12, 512]

# Add positional encoding
pos_encoder = PositionalEncoding(embedding_dim, max_len=12)
x = pos_encoder(embeddings)

# Create Transformer Encoder Layer
encoder_layer = TransformerEncoderLayer(embedding_dim=256, num_heads=4, hidden_dim=1024)

# Forward pass
output = encoder_layer(x)

# Print actual tensor output values (for first token and first 5 dimensions for readability)
print("Actual output tensor (first token, first 5 dims):")
print(output[0, :, :256].detach().numpy())

# Optional: To print entire tensor output (warning: large output)
# print(output.detach().numpy())