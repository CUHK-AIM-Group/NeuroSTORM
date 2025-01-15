import torch
import torch.nn as nn


class ref_head_v1(nn.Module):
    def __init__(self, num_tokens = 96):
        super(ref_head_v1, self).__init__()
        num_outputs = 1
        self.head = nn.Linear(num_tokens, num_outputs)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x -> (b, 96, 4, 4, 4, t)
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class reg_head(nn.Module):
    def __init__(self, version=1, num_tokens=96):
        super(reg_head, self).__init__()
        if version == 1:
            self.head = ref_head_v1(num_tokens)

    def forward(self, x):
        return self.head(x)