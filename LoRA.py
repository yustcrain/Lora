import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, base):
        super().__init__()

        self.base = base

        for parm in self.base.parameters():
            parm.requires_grad(False)
        
        self.in_feature = in_features
        self.out_feature = out_features

        self.A = nn.Linear(self.in_feature, rank)
        self.B = nn.Linear(self.out_feature, rank)

        self.A.weight.normal_(mean=0, std=0.2)
        self.B.weight.zero_()
        self.scale = alpha / rank



    def forward(self, x):
        base_out = self.base(x)
        return base_out+self.B(self.A(x)) * self.scale

    def to_merge_linear(self):
        merged = nn.Linear(
                    self.base.in_features,
                    self.base.out_features,
                    bias=self.base.bias is not None,
                ).to(
                    device=self.base.weight.device,
                    dtype=self.base.weight.dtype,
                )

        merged.weight.copy_(
            self.base.weight + self.scaling * (self.B @ self.A)
        )

        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)

        merged.requires_grad_(False)
        return merged.eval()
    
