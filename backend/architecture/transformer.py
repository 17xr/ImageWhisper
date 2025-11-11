import torch
from torch import nn
from torch.nn.functional import silu
from transformers import AutoModel


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

        angle_rates = 1.0 / torch.pow(
            10_000, torch.arange(0, embed_dim, 2).float() / embed_dim
        )
        angles = torch.arange(max_seq_len).unsqueeze(1) * angle_rates.unsqueeze(0)

        self.register_buffer("sin", angles.sin(), persistent=False)
        self.register_buffer("cos", angles.cos(), persistent=False)

    def forward(self, x):
        seq_len = x.shape[1]

        sin = self.sin[None, :seq_len, None, :]
        cos = self.cos[None, :seq_len, None, :]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        embeddings = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return embeddings.flatten(-2)


class BaseMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super(BaseMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim**-0.5

        self.pre_norm = nn.RMSNorm(embed_dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.rope = RotaryPositionalEmbeddings(max_seq_len, self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _prepare_qkv(self, q, k, v):
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.head_dim)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.head_dim)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)
        return q, k, v

    def _apply_mask(self, q, k, attn_mask, padding_mask):
        alignment = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            alignment = alignment.masked_fill(attn_mask == 0, float("-inf"))

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            alignment = alignment.masked_fill(padding_mask == 0, float("-inf"))

        return alignment

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward(): Not implemented")


class SelfMultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super(SelfMultiHeadAttention, self).__init__(
            embed_dim, num_heads, max_seq_len, dropout
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _project_qkv(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        return q, k, v

    def forward(self, x, attn_mask=None, padding_mask=None):
        batch_size, seq_length, embed_dim = x.shape

        x_norm = self.pre_norm(x)
        q, k, v = self._project_qkv(x_norm)
        q, k, v = self._prepare_qkv(q, k, v)

        alignment = self._apply_mask(q, k, attn_mask, padding_mask)
        attention = torch.softmax(alignment, dim=-1)
        context = torch.matmul(self.dropout(attention), v)
        out = context.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        return x + self.proj(out)


class CrossMultiHeadAttention(BaseMultiHeadAttention):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super(CrossMultiHeadAttention, self).__init__(
            embed_dim, num_heads, max_seq_len, dropout
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, k, v, attn_mask=None, padding_mask=None):
        batch_size, seq_length, embed_dim = x.shape

        x_norm = self.pre_norm(x)
        q = self.q_proj(x_norm)
        q, k, v = self._prepare_qkv(q, k, v)

        alignment = self._apply_mask(q, k, attn_mask, padding_mask)
        attention = torch.softmax(alignment, dim=-1)
        context = torch.matmul(self.dropout(attention), v)
        out = context.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        return x + self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()

        self.pre_norm = nn.RMSNorm(embed_dim)

        self.feature = nn.Linear(embed_dim, hidden_dim)
        self.gate = nn.Linear(embed_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.pre_norm(x)
        feature = silu(self.feature(x_norm))
        gate = self.gate(x_norm)
        output = x + self.proj(feature * gate)
        return self.dropout(output)


class DecoderTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super(DecoderTransformerBlock, self).__init__()

        self.self_attention = SelfMultiHeadAttention(
            embed_dim, num_heads, max_seq_len, dropout
        )
        self.cross_attention = CrossMultiHeadAttention(
            embed_dim, num_heads, max_seq_len, dropout
        )
        self.feed_forward = FeedForward(embed_dim, embed_dim * 3, dropout)

    def forward(self, x, k, v, attn_mask=None, padding_mask=None):
        self_attention = self.self_attention(x, attn_mask, padding_mask)
        cross_attention = self.cross_attention(self_attention, k, v)
        feed_forward = self.feed_forward(cross_attention)

        return feed_forward


class DecoderTransformer(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, max_seq_len, depth, dropout=0.1
    ):
        super(DecoderTransformer, self).__init__()

        assert embed_dim % num_heads == 0

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderTransformerBlock(embed_dim, num_heads, max_seq_len, dropout)
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
        self,
        device,
        text_embeddings,
        tokenizer,
        embed_dim,
        num_heads,
        max_seq_len,
        depth,
        dropout=0.1,
    ):
        super(ImageCaption, self).__init__()

        self.text_embeddings = text_embeddings
        self.vision_encoder = ImageEncoder(embed_dim)
        self.text_transformer = DecoderTransformer(
            tokenizer.vocab_size, embed_dim, num_heads, max_seq_len, depth, dropout
        )

        self.device = device

    def forward(self, pixel_values, input_tokens, attn_mask=None, padding_mask=None):
        k, v = self.vision_encoder(pixel_values)
        embedded_tokens = self.text_embeddings(input_tokens)

        logits = self.text_transformer(embedded_tokens, k, v, attn_mask, padding_mask)
        return logits
