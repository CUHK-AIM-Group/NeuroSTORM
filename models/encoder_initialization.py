"""Load complete encoder weights while leaving a new supervised head intact."""

from collections.abc import Mapping

import torch


def initialize_encoder(encoder, checkpoint_path, prefixes):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a state dictionary or a checkpoint containing one")
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint state must be a mapping")
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        while key.startswith(("module.", "model.")):
            key = key.split(".", 1)[1]
        normalized[key] = value
    expected = set(encoder.state_dict())
    candidates = [
        {key[len(prefix):]: value for key, value in normalized.items()
         if key.startswith(prefix) and key[len(prefix):] in expected}
        for prefix in prefixes
    ]
    selected = max(candidates, key=len)
    # strict=True rejects partial encoders and incompatible shapes. Decoder and
    # old task-head weights are deliberately outside the encoder key set.
    encoder.load_state_dict(selected, strict=True)

