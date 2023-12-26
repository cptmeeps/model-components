import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
  def __init__(self, input_vocab_size, output_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
    super(TransformerModel, self).__init__()
    self.input_embedding = nn.Embedding(input_vocab_size, d_model)
    self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
    self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)])
    self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)])
    self.output_linear = nn.Linear(d_model, output_vocab_size)
    self.output_softmax = nn.LogSoftmax(dim=-1)

  def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
    src_emb = self.positional_encoding(self.input_embedding(src))
    memory = src_emb
    for layer in self.encoder_layers:
      memory = layer(memory, src_mask, src_padding_mask)

    tgt_emb = self.positional_encoding(self.input_embedding(tgt))
    output = tgt_emb
    for layer in self.decoder_layers:
      output = layer(output, memory, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)

    output = self.output_linear(output)
    output = self.output_softmax(output)
    return output

class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout):
    super(TransformerEncoderLayer, self).__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, src, src_mask, src_key_padding_mask):
    src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
    src = src + self.dropout(src2)
    src = self.norm1(src)
    src2 = self.feed_forward(src)
    src = src + self.dropout(src2)
    src = self.norm2(src)
    return src

class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout):
    super(TransformerDecoderLayer, self).__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
    tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
    tgt = tgt + self.dropout(tgt2)
    tgt = self.norm1(tgt)
    tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.feed_forward(tgt)
    tgt = tgt + self.dropout(tgt2)
    tgt = self.norm3(tgt)
    return tgt

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout, max_len):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)
  
class SwiGLU(nn.Module):
  def forward(self, x):
    dim = x.shape[-1] // 2
    return x[..., :dim] * F.silu(x[..., dim:])

class FeedForward(nn.Module):
  def __init__(self, d_model, dim_feedforward, dropout):
    super(FeedForward, self).__init__()
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

  def forward(self, x):
    return self.linear2(self.dropout(SwiGLU(self.linear1(x))))
