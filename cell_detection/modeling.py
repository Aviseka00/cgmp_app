from __future__ import annotations

import segmentation_models_pytorch as smp


def build_model(encoder: str = "resnet34") -> smp.Unet:
    """ImageNet-pretrained U-Net for binary positive-cell segmentation."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
