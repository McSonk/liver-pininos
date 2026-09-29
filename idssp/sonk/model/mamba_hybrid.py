"""
Minimal viable 2.5D Mamba-hybrid module for integration testing.

This MVP implements only the `use_z_context=False` ablation path:

    input volume -> axial slices -> 2D encoder -> bottleneck -> 2D decoder
    -> axial logits -> restored volume

Stages 3 to 8 from the settled design table are intentionally absent in this
iteration. The next iteration should scaffold the `use_z_context=True` path,
initially with an identity or lightweight placeholder in place of Mamba2, so
that the sequence reshape logic can be exercised inside the real model.
"""

import torch
import torch.nn as nn

from idssp.sonk.model.cnn2d import Decoder2D, Encoder2D
from idssp.sonk.model.mamba_axis import merge_axial_slices, split_into_axial_slices


class MambaHybrid(nn.Module):
    """
    Minimal viable 2.5D Mamba-hybrid model.

    External tensor contract:

        input:  (B, C, X, Y, Z)
        output: (B, NUM_CLASSES, X, Y, Z)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 16,
        num_downs: int = 4,
        use_z_context: bool = False,
        norm_type: str = "group",
        norm_num_groups: int = 8,
    ) -> None:
        """
        Parameters
        ----------
        in_channels:
            Number of input channels. For the current LiTS CT pipeline this is 1.
        num_classes:
            Number of segmentation classes, including background.
        base_channels:
            Base channel width of the 2D encoder/decoder.
        num_downs:
            Number of 2D downsampling steps.
        use_z_context:
            Ablation flag for the z-context path. The final model is expected to
            default this to True once the Mamba path is implemented. In this MVP,
            only False is supported.
        norm_type:
            Normalisation type used by the 2D CNN blocks.
        norm_num_groups:
            Number of groups for GroupNorm.
        """
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}.")
        if num_downs < 1:
            raise ValueError(f"num_downs must be at least 1, got {num_downs}.")

        if use_z_context:
            raise NotImplementedError(
                "MambaHybrid MVP implements only use_z_context=False. "
                "Stages 3 to 8 will be scaffolded in the next iteration."
            )

        self.use_z_context = use_z_context

        self.encoder = Encoder2D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_downs=num_downs,
            norm_type=norm_type,
            num_groups=norm_num_groups,
        )
        self.decoder = Decoder2D(
            num_classes=num_classes,
            base_channels=base_channels,
            num_downs=num_downs,
            norm_type=norm_type,
            num_groups=norm_num_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor with shape `(B, C, X, Y, Z)`.

        Returns
        -------
        Tensor with shape `(B, NUM_CLASSES, X, Y, Z)`.
        """
        if self.use_z_context:
            raise NotImplementedError(
                "MambaHybrid MVP implements only use_z_context=False. "
                "Stages 3 to 8 will be scaffolded in the next iteration."
            )

        slices, meta = split_into_axial_slices(x)

        bottleneck, skips = self.encoder(slices)

        # MVP bypasses Stages 3 to 8.
        # The bottleneck is fed directly to the decoder, matching the
        # use_z_context=False ablation path in the settled design.
        logits = self.decoder(bottleneck, skips)

        return merge_axial_slices(logits, meta)
