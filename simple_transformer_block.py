import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEncoding(nn.Module):
  def __init__(self, dim, base=10000):
    super().__init__()
    inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)

  def forward(self, max_seq_len):
    t = torch.arange(max_seq_len).type_as(self.inv_freq)
    freqs = torch.einsum('i,j->ij', t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.stack((emb.sin(), emb.cos()), dim=-1).view(max_seq_len, -1)

def apply_rotary_pos_emb(x, sincos):
  sin, cos = map(lambda t: t.repeat_interleave(2, dim=-1), sincos.chunk(2, dim=-1))
  return (x * cos) + (x.flip(dims=(-1,)) * sin)

class SwiGLU(nn.Module):
  def forward(self, x):
    dim = x.shape[-1] // 2
    return x[..., :dim] * F.silu(x[..., dim:])

class Encoder(nn.Module):
  def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
    super(Encoder, self).__init__()
    self.embed_size = embed_size
    self.device = device
    self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
    self.positional_embedding = nn.Embedding(max_length, embed_size)

    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          embed_size,
          heads,
          dropout=dropout,
          forward_expansion=forward_expansion,
        )
        for _ in range(num_layers)
      ]
    )

    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    N, seq_length = x.shape
    positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
    out = self.dropout(
      (self.word_embedding(x) + self.positional_embedding(positions))
    )

    for layer in self.layers:
      out = layer(out, out, out, mask)

    return out

class Decoder(nn.Module):
  def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
    super(Decoder, self).__init__()
    self.embed_size = embed_size
    self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
    self.positional_embedding = nn.Embedding(max_length, embed_size)

    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          embed_size,
          heads,
          dropout=dropout,
          forward_expansion=forward_expansion,
        )
        for _ in range(num_layers)
      ]
    )

    self.dropout = nn.Dropout(dropout)
    self.output_layer = nn.Linear(embed_size, trg_vocab_size)

  def forward(self, x, enc_out, src_mask, trg_mask):
    N, seq_length = x.shape
    positions = torch.arange(0, seq_length).expand(N, seq_length)
    x = self.dropout(
      (self.word_embedding(x) + self.positional_embedding(positions))
    )

    for layer in self.layers:
      x = layer(x, enc_out, x, trg_mask)

    out = self.output_layer(x)
    return out

class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_size, forward_expansion * embed_size),
      SwiGLU(),
      nn.Linear(forward_expansion * embed_size, embed_size)
    )

    self.dropout = nn.Dropout(dropout)
    self.rotary_pos_emb = RotaryPositionalEncoding(embed_size // 2)

  def forward(self, value, key, query, mask):
    max_seq_len = query.shape[1]
    sincos = self.rotary_pos_emb(max_seq_len).to(query.device)
    query, key, value = map(lambda t: apply_rotary_pos_emb(t, sincos), (query, key, value))

    query_norm = self.norm1(query)
    attention = self.attention(query_norm, key, value, attn_mask=mask)[0]
    x = self.dropout(attention) + query
    x_norm = self.norm2(x)
    forward = self.feed_forward(x_norm)
    out = self.dropout(forward) + x
    return out


class Transformer(nn.Module):
  def __init__(self, src_vocab_size, trg_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
    super(Transformer, self).__init__()
    self.encoder = Encoder(
      src_vocab_size,
      embed_size,
      num_layers,
      heads,
      device,
      forward_expansion,
      dropout,
      max_length
    )
    self.decoder = Decoder(
      trg_vocab_size,
      embed_size,
      num_layers,
      heads,
      forward_expansion,
      dropout,
      max_length
    )

  def forward(self, src, trg, src_mask, trg_mask):
    enc_src = self.encoder(src, src_mask)
    out = self.decoder(trg, enc_src, src_mask, trg_mask)
    return out
