"""Final-layer BrainLM readouts and the checkpoint's trained scan metric."""
import torch
from torch import nn
import torch.nn.functional as F
from .brainlm_modeling import ViTMAEModel


class BrainReadout(nn.Module):
    def __init__(self, config, readout):
        super().__init__()
        if readout not in ("cls", "foreground_mean", "cls_mean_blend"):
            raise ValueError(readout)
        config.train_mode = "auto_encode"
        config.mask_ratio = 0.
        self.backbone = ViTMAEModel(config)
        self.classifier = nn.Linear(config.hidden_size, 867)
        self.readout = readout
        weights = F.avg_pool2d(F.pad(torch.ones(1, 1, 424, 200), (116, 116, 4, 4)), 16).flatten()
        self.register_buffer("foreground_weights", weights, persistent=False)

    def forward(self, image):
        image = F.pad(image, (116, 116, 4, 4), value=-1)
        output = self.backbone(pixel_values=image)
        cls = output.last_hidden_state[:, 0]
        if self.readout == "cls":
            z = cls
        else:
            patch = torch.gather(output.last_hidden_state[:, 1:], 1,
                                 output.ids_restore[:, :, None].expand_as(output.last_hidden_state[:, 1:]))
            mean = (patch * self.foreground_weights[None, :, None]).sum(1) / self.foreground_weights.sum()
            z = mean if self.readout == "foreground_mean" else .5 * (F.normalize(cls, dim=1) + F.normalize(mean, dim=1))
        return self.classifier(z), z


class RecoveryMetricBrain(nn.Module):
    def __init__(self, config, mean, matrix, readout):
        super().__init__()
        if (config.hidden_size, config.num_hidden_layers) != (768, 12):
            raise ValueError("The submitted BrainLM checkpoint uses the 12-layer encoder")
        self.encoder = BrainReadout(config, readout)
        self.register_buffer("metric_mean", torch.as_tensor(mean, dtype=torch.float64))
        self.register_buffer("metric_matrix", torch.as_tensor(matrix, dtype=torch.float64))

    def encode_windows(self, image):
        return self.encoder(image)
