import torch
import torch.nn as nn
import math


class EncoderBlock(nn.Module):

    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout):
        super().__init__()

        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Initialize MLP weights
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x):
        x1 = self.ln_1(x)
        attn, _ = self.self_attn(x1, x1, x1, need_weights=False)
        x = x + self.dropout(attn)

        x2 = self.ln_2(x)
        x = x + self.mlp(x2)

        return x


class Encoder(nn.Module):

    def __init__(self, seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.Sequential(*[EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x):
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.layers(x)
        x = self.ln(x)

        return x


class VisionTransformer(nn.Module):

    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout=0.0, attention_dropout=0.0, num_classes=100):
        super().__init__()

        assert image_size % patch_size == 0, "Input shape indivisible by patch size!"
        self.image_size = image_size

        self.conv_proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder = Encoder((image_size // patch_size)**2 + 1, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

        # Initialize weights for the convolutional projection and the classification head
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / (3 * patch_size ** 2)))
        nn.init.zeros_(self.conv_proj.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, images):
        assert images.shape[2] == self.image_size, f"Expected image height {self.image_size}"
        assert images.shape[3] == self.image_size, f"Expected image width {self.image_size}"

        x = self.conv_proj(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        x = self.encoder(x)
        x = self.head(x[:, 0])

        return x


def vit_s_16(*, image_size=224, **kwargs):
    return VisionTransformer(
        image_size=image_size,
        patch_size=16,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=4*384,
        **kwargs,
    )


def vit_t_16(*, image_size=224, **kwargs):
    return VisionTransformer(
        image_size=image_size,
        patch_size=16,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=4*192,
        **kwargs,
    )
