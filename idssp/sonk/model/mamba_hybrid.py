"""
2.5D Mamba-hybrid model with identity z-context scaffold.

External tensor contract:

    input:  (B, C, X, Y, Z)
    output: (B, NUM_CLASSES, X, Y, Z)

Current status:

    use_z_context=False:
        Ablation path. The encoder bottleneck is fed directly to the decoder.

    use_z_context=True:
        Identity scaffold path. Stages 3, 4, 6, 7a, 7b, and 8 from the settled
        design table are implemented. Stage 5 is currently an `nn.Identity()`
        placeholder. It will be replaced by `mamba_ssm.Mamba2(d_model=C_bot)`
        in the next CUDA-only integration step.

This module remains CPU-safe and does not import `mamba_ssm`.
"""

import torch
import torch.nn as nn

from idssp.sonk.model.cnn2d import Decoder2D, Encoder2D
from idssp.sonk.model.mamba_axis import merge_axial_slices, split_into_axial_slices


class MambaHybrid(nn.Module):
    """
    Minimal 2.5D Mamba-hybrid model.

    The decoder input channel count is always `C_bot`, regardless of whether
    `use_z_context` is True or False. This keeps the ablation path decoder
    contract aligned with the z-context path.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 16,
        num_downs: int = 4,
        use_z_context: bool = True,
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
            If True, enable the z-context scaffold path. If False, bypass
            Stages 3 to 8 and feed the bottleneck directly to the decoder.
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

        self.use_z_context = use_z_context
        self.bottleneck_channels = base_channels * (2 ** num_downs)

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

        if use_z_context:
            # TODO: Replace nn.Identity() with mamba_ssm.Mamba2(d_model=self.bottleneck_channels)
            # on the server inside ~/mamba-env after verifying the installed Mamba2 signature.
            # Do not import mamba_ssm in this CPU-safe module.
            self.z_context = nn.Identity()
            self.fusion_conv = nn.Conv2d(
                in_channels=2 * self.bottleneck_channels,
                out_channels=self.bottleneck_channels,
                kernel_size=1,
                bias=True,
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
        # Stage 1: split volume into axial slices.
        slices, meta = split_into_axial_slices(x)

        # Stage 2: 2D encoder down path.
        bottleneck, skips = self.encoder(slices)

        if self.use_z_context:
            rows, channels, height, width = bottleneck.shape

            if channels != self.bottleneck_channels:
                raise RuntimeError(
                    "Encoder bottleneck channel count does not match MambaHybrid. "
                    f"Expected {self.bottleneck_channels}, got {channels}."
                )

            # Stage 3: global average pool each slice's spatial bottleneck.
            pooled = bottleneck.mean(dim=(2, 3))  # (B*Z, C_bot)

            # Stage 4: un-merge to sequence form.
            sequence = pooled.reshape(meta.batch_size, meta.z, channels)  # (B, Z, C_bot)

            # Stage 5: z-context model.
            # Identity placeholder for now; later Mamba2.
            sequence = self.z_context(sequence)  # (B, Z, C_bot)

            # Stage 6: re-merge B·Z.
            z_flat = sequence.reshape(rows, channels)  # (B*Z, C_bot)

            # Stage 7a: broadcast z-context spatially.
            z_spatial = z_flat.unsqueeze(-1).unsqueeze(-1).expand(
                rows,
                channels,
                height,
                width,
            )  # (B*Z, C_bot, H_bot, W_bot)

            # Stage 7b: concatenate with cached bottleneck.
            fused = torch.cat((bottleneck, z_spatial), dim=1)  # (B*Z, 2*C_bot, H_bot, W_bot)

            # Stage 8: 1x1 fusion convolution.
            decoder_input = self.fusion_conv(fused)  # (B*Z, C_bot, H_bot, W_bot)
        else:
            # Ablation path: bypass Stages 3 to 8.
            decoder_input = bottleneck

        # Stage 9: 2D decoder up path.
        logits = self.decoder(decoder_input, skips)

        # Stage 10 is included in Decoder2D as the final 1x1 head.
        # Stage 11: restore 3D volume.
        return merge_axial_slices(logits, meta)
