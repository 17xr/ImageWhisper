import torch.nn.functional as F
import torch
from transformers import AutoModel
from torch import nn


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.RMSNorm):
        nn.init.constant_(module.weight, 1.0)


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(ImageEncoder, self).__init__()

        model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = self.backbone.config.hidden_size
        self.k_proj = nn.Linear(hidden_size, embed_dim, bias=False)
        self.k_norm = nn.RMSNorm(embed_dim)
        self.v_proj = nn.Linear(hidden_size, embed_dim, bias=False)
        self.v_norm = nn.RMSNorm(embed_dim)

        _init_weights(self.k_proj)
        _init_weights(self.k_norm)
        _init_weights(self.v_proj)
        _init_weights(self.v_norm)

    def forward(self, x):
        outputs = self.backbone(x)
        patch_embeddings = outputs.last_hidden_state[:, 5:, :]

        k_proj = self.k_norm(self.k_proj(patch_embeddings))
        v_proj = self.v_norm(self.v_proj(patch_embeddings))
        return k_proj, v_proj


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        super(RotaryPositionalEmbeddings, self).__init__()

        inv_freq = 1.0 / (10_000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)

        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):
        seq_len = x.shape[-2]

        cos = self.cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin[:seq_len].view(1, 1, seq_len, -1)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        x_out1 = x1 * cos - x2 * sin
        x_out2 = x1 * sin + x2 * cos

        return torch.stack((x_out1, x_out2), dim=-1).flatten(-2)


class BaseMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_length):
        super(BaseMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.pre_norm = nn.RMSNorm(embed_dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.rope = RotaryPositionalEmbeddings(max_length, self.head_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _prepare_qkv(self, q, k, v):
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)
        return q, k, v

    def _compute_attention(self, q, k, v, attn_mask=None, padding_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        return context


class SelfMultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, embed_dim, num_heads, max_length):
        super(SelfMultiHeadAttention, self).__init__(embed_dim, num_heads, max_length)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, attn_mask=None, padding_mask=None):
        x_norm = self.pre_norm(x)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        q, k, v = self._prepare_qkv(q, k, v)
        context = self._compute_attention(q, k, v, attn_mask, padding_mask)

        out = context.transpose(1, 2).contiguous().view(*x.shape)
        return x + self.proj(out)


class CrossMultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, embed_dim, num_heads, max_length):
        super(CrossMultiHeadAttention, self).__init__(embed_dim, num_heads, max_length)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, k, v):
        x_norm = self.pre_norm(x)

        q = self.q_proj(x_norm)
        q, k, v = self._prepare_qkv(q, k, v)

        context = self._compute_attention(q, k, v)

        out = context.transpose(1, 2).contiguous().view(*x.shape)
        return x + self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()

        self.pre_norm = nn.RMSNorm(embed_dim)
        self.feature = nn.Linear(embed_dim, hidden_dim)
        self.gate = nn.Linear(embed_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x_norm = self.pre_norm(x)
        feature = F.silu(self.feature(x_norm))
        gate = self.gate(x_norm)
        return x + self.proj(feature * gate)


class DecoderTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_length):
        super(DecoderTransformerBlock, self).__init__()

        self.self_attention = SelfMultiHeadAttention(embed_dim, num_heads, max_length)
        self.cross_attention = CrossMultiHeadAttention(embed_dim, num_heads, max_length)
        self.feed_forward = FeedForward(embed_dim, embed_dim * 3)

    def forward(self, x, k, v, attn_mask=None, padding_mask=None):
        x = self.self_attention(x, attn_mask, padding_mask)
        x = self.cross_attention(x, k, v)
        x = self.feed_forward(x)
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, max_length, depth):
        super(DecoderTransformer, self).__init__()

        assert embed_dim % num_heads == 0

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderTransformerBlock(embed_dim, num_heads, max_length)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.RMSNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

        self.apply(_init_weights)

    def forward(self, x, k, v, attn_mask=None, padding_mask=None):
        for block in self.decoder_blocks:
            x = block(x, k, v, attn_mask, padding_mask)
        return self.output(self.final_norm(x))


class ImageCaption(nn.Module):
    def __init__(
        self, text_embeddings, vocab_size, embed_dim, num_heads, max_length, depth
    ):
        super(ImageCaption, self).__init__()

        self.text_embeddings = text_embeddings
        self.vision_encoder = ImageEncoder(embed_dim)
        self.text_transformer = DecoderTransformer(
            vocab_size, embed_dim, num_heads, max_length, depth
        )

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_length, max_length, dtype=torch.bool)),
        )

    def forward(self, pixel_values, input_tokens, padding_mask=None):
        k, v = self.vision_encoder(pixel_values)
        embedded_tokens = self.text_embeddings(input_tokens)

        seq_len = input_tokens.shape[1]
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        logits = self.text_transformer(embedded_tokens, k, v, attn_mask, padding_mask)
        return logits
