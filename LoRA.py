import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        self.in_feature = in_features
        self.out_feature = out_features

        self.A = nn.Linear(self.in_feature, rank)
        self.B = nn.Linear(self.out_feature, rank)

        self.A.weight.normal_(mean=0, std=0.2)
        self.B.weight.zero_()
        self.scale = alpha / rank

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


def apply_lora(model, rank, alpha):
    for name, module in model.named_modules():
        if isinstance(model, nn.Linear) and model.in_features == model.out_features:
            lora = LoRA(in_features=model.in_features, out_features=model.out_features, rank=rank, alpha=alpha)
            setattr(module, "lora", lora)
            model.requires_grad_(False)
            original_forward = module.forward.requires_grad_

            def forward_with_lora(x, layer1, layer2):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
