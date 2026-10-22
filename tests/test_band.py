import torch

from models.band import BAnD, ResNet3D18Encoder, sinusoidal_position_encoding


def test_position_encoding_shape_and_origin():
    encoding = sinusoidal_position_encoding(5, 64)
    assert encoding.shape == (1, 5, 64)
    assert torch.allclose(encoding[0, 0, 0::2], torch.zeros(32))
    assert torch.allclose(encoding[0, 0, 1::2], torch.ones(32))


def test_frame_encoder_shape():
    model = ResNet3D18Encoder(in_channels=1, base_channels=8).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 1, 32, 32, 32))
    assert output.shape == (2, 64)


def test_band_forward_and_feature_shape():
    model = BAnD(
        num_classes=7,
        base_channels=8,
        transformer_layers=2,
        num_heads=8,
        feedforward_dim=128,
        frame_chunk_size=2,
        max_sequence_length=4,
    ).eval()
    fmri = torch.randn(2, 1, 32, 32, 32, 4)
    with torch.no_grad():
        logits, features = model.forward_with_features(fmri)
    assert logits.shape == (2, 7)
    assert features.shape == (2, 64)
