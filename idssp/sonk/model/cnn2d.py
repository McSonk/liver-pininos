"""
Minimal 2D CNN components for the 2.5D Mamba-hybrid MVP.

This module is intentionally a straightforward feature extractor and is not the
research contribution. It provides a conventional 2D encoder/decoder for axial
slices produced by `mamba_axis.split_into_axial_slices`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Run normalisation experiments before choosing the production setting.
# Compare at least:
#   - GroupNorm
#   - InstanceNorm2d
#   - BatchNorm2d
# Metrics to compare:
#   - Training stability: training loss, validation loss, liver Dice, tumour Dice
#   - Convergence: whether one option reaches useful performance substantially earlier
#   - Run-to-run variance: reproducibility across repeated runs
def _make_norm(norm_type: str, channels: int, num_groups: int) -> nn.Module:
    """
    Create a normalisation layer for 2D convolutional blocks.

    Parameters
    ----------
    norm_type:
        One of {"group", "instance", "batch"}.
    channels:
        Number of channels of the tensor that will be normalised.
    num_groups:
        Number of groups for GroupNorm.
    """
    norm_type = norm_type.lower().strip()

    if norm_type == "group":
        if channels % num_groups != 0:
            raise ValueError(
                "GroupNorm requires the channel count to be divisible by num_groups. "
                f"Got channels={channels}, num_groups={num_groups}."
            )
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    if norm_type == "instance":
        return nn.InstanceNorm2d(num_features=channels, affine=True)

    if norm_type == "batch":
        return nn.BatchNorm2d(num_features=channels)

    raise ValueError(
        f"Unsupported norm_type '{norm_type}'. Expected one of: 'group', 'instance', 'batch'."
    )


class ConvNormAct(nn.Module):
    """Conv2d -> Norm -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group",
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm = _make_norm(norm_type=norm_type, channels=out_channels, num_groups=num_groups)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    """Two consecutive Conv-Norm-Act blocks, as in a conventional UNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group",
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_type=norm_type,
                num_groups=num_groups,
            ),
            ConvNormAct(
                in_channels=out_channels,
                out_channels=out_channels,
                norm_type=norm_type,
                num_groups=num_groups,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder2D(nn.Module):
    """
    Minimal 2D encoder for axial slices.

    For `base_channels=16` and `num_downs=4`, the channel schedule is:

        16 -> 32 -> 64 -> 128 -> 256

    The first four resolutions are returned as skip connections. The final
    256-channel feature map is the bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        num_downs: int = 4,
        norm_type: str = "group",
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}.")
        if num_downs < 1:
            raise ValueError(f"num_downs must be at least 1, got {num_downs}.")

        self.num_downs = num_downs
        self._spatial_divisor = 2 ** num_downs
        self.channels = [base_channels * (2 ** i) for i in range(num_downs + 1)]

        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for out_channels in self.channels[:-1]:
            self.blocks.append(
                DoubleConv(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )
            current_channels = out_channels

        self.bottleneck = DoubleConv(
            in_channels=current_channels,
            out_channels=self.channels[-1],
            norm_type=norm_type,
            num_groups=num_groups,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Parameters
        ----------
        x:
            Tensor with shape `(rows, C, X, Y)`.

        Returns
        -------
        bottleneck:
            Tensor with shape `(rows, C_bot, X', Y')`.
        skips:
            Skip tensors from the four higher resolutions, ordered from
            highest resolution to lowest resolution.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, got shape {tuple(x.shape)}.")

        if x.shape[-2] % self._spatial_divisor != 0 or x.shape[-1] % self._spatial_divisor != 0:
            raise ValueError(
                "Spatial dimensions must be divisible by "
                f"{self._spatial_divisor} for {self.num_downs} downsampling steps. "
                f"Got spatial shape {tuple(x.shape[-2:])}."
            )

        skips: list[torch.Tensor] = []

        for block in self.blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        bottleneck = self.bottleneck(x)
        return bottleneck, skips


class UpBlock(nn.Module):
    """
    Upsample to the skip spatial size, concatenate the skip, then apply a
    DoubleConv block.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_type: str = "group",
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.conv = DoubleConv(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            norm_type=norm_type,
            num_groups=num_groups,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class Decoder2D(nn.Module):
    """
    Minimal 2D decoder consuming the encoder skip connections.

    For `base_channels=16` and `num_downs=4`, the decoder channel schedule is:

        256 -> 128 -> 64 -> 32 -> 16 -> NUM_CLASSES
    """

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 16,
        num_downs: int = 4,
        norm_type: str = "group",
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}.")
        if num_downs < 1:
            raise ValueError(f"num_downs must be at least 1, got {num_downs}.")

        channels = [base_channels * (2 ** i) for i in range(num_downs + 1)]

        self.up_blocks = nn.ModuleList()
        current_channels = channels[-1]

        for skip_channels in reversed(channels[:-1]):
            self.up_blocks.append(
                UpBlock(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=skip_channels,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )
            current_channels = skip_channels

        self.head = nn.Conv2d(
            in_channels=current_channels,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        bottleneck:
            Tensor with shape `(rows, C_bot, X', Y')`.
        skips:
            Skip tensors from Encoder2D, ordered from highest resolution to
            lowest resolution.

        Returns
        -------
        logits:
            Tensor with shape `(rows, NUM_CLASSES, X, Y)`.
        """
        if len(skips) != len(self.up_blocks):
            raise ValueError(
                "Skip connection count does not match decoder upsampling blocks. "
                f"Expected {len(self.up_blocks)} skips, got {len(skips)}."
            )

        x = bottleneck
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)

        return self.head(x)
