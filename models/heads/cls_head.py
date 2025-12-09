import torch
import torch.nn as nn


class cls_head_v1(nn.Module):
    def __init__(self, num_classes=2, num_tokens = 96):
        super(cls_head_v1, self).__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.head = nn.Linear(num_tokens, num_outputs)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(num_tokens)

    def forward(self, x):
        # x -> (b, 96, 4, 4, 4, t)
        # torch.Size([16, 288, 2, 2, 2, 20])
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # torch.Size([16, 160, 288])
        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        # torch.Size([16, 288, 1])
        x = torch.flatten(x, 1)
        # torch.Size([16, 288])
        x = self.head(x)
        # torch.Size([16, 1])
        return x

    def forward_with_features(self, x):
        x = x.flatten(start_dim=2).transpose(1, 2)
        pooled = self.avgpool(x.transpose(1, 2))
        feat = torch.flatten(pooled, 1)
        logits = self.head(feat)
        return logits, feat
    

class cls_head_v2(nn.Module):
    def __init__(self, num_classes=2, num_tokens = 96):
        super(cls_head_v2, self).__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.hidden = nn.Linear(num_tokens, 4*num_tokens)
        self.head = nn.Linear(4*num_tokens, num_outputs)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x -> (b, 96, 4, 4, 4, t)
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.hidden(x)
        x = self.head(x)

        return x

    def forward_with_features(self, x):
        x = x.flatten(start_dim=2).transpose(1, 2)
        pooled = self.avgpool(x.transpose(1, 2))
        pooled = torch.flatten(pooled, 1)
        hidden = self.hidden(pooled)
        logits = self.head(hidden)
        return logits, hidden


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=288, num_heads=8, num_layers=6, forward_expansion=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_size, 
                nhead=num_heads, 
                dim_feedforward=emb_size * forward_expansion, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class cls_head_v3(nn.Module):
    def __init__(self, num_classes, num_tokens=160, emb_size=288, num_heads=8, num_layers=6, forward_expansion=4, dropout=0.1):
        super().__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, emb_size))
        self.transformer_encoder = TransformerEncoder(emb_size=emb_size, num_heads=num_heads, num_layers=num_layers, forward_expansion=forward_expansion, dropout=dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_outputs)
        )

    def forward(self, x):
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        B, L, C = x.shape
        assert L == self.pos_embed.shape[1] - 1, f"Expected input with {self.pos_embed.shape[1] - 1} tokens, but got {L} tokens."
        assert C == self.cls_token.shape[2], f"Expected input embedding size of {self.cls_token.shape[2]}, but got {C}."
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :L+1, :]
        x = self.transformer_encoder(x)
        cls_token_final = x[:, 0]
        x = self.mlp_head(cls_token_final)
        
        return x

    def forward_with_features(self, x):
        x = x.flatten(start_dim=2).transpose(1, 2)
        B, L, C = x.shape
        assert L == self.pos_embed.shape[1] - 1, f"Expected input with {self.pos_embed.shape[1] - 1} tokens, but got {L} tokens."
        assert C == self.cls_token.shape[2], f"Expected input embedding size of {self.cls_token.shape[2]}, but got {C}."

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :L+1, :]
        x = self.transformer_encoder(x)
        cls_token_final = x[:, 0]
        feat = self.mlp_head[0](cls_token_final)
        logits = self.mlp_head[1](feat)
        return logits, feat


class cls_head(nn.Module):
    def __init__(self, version=1, num_classes=2, num_tokens=96):
        super(cls_head, self).__init__()
        if version == 'v1':
            self.head = cls_head_v1(num_classes, num_tokens)
        elif version == 'v2':
            self.head = cls_head_v2(num_classes, num_tokens)
        elif version == 'v3':
            self.head = cls_head_v3(num_classes, num_tokens)

    def forward(self, x):
        return self.head(x)

    def forward_with_features(self, x):
        if hasattr(self.head, "forward_with_features"):
            return self.head.forward_with_features(x)
        logits = self.head(x)
        return logits, None