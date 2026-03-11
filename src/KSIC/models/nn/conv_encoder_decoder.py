from typing import List, Tuple
import torch
from torch import Tensor
from torch import nn


class ConvEncoder(nn.Module):
    """
    Encodeur convolutionnel :
    (B, C, H, W) -> (B, D_latent)
    - out_planes, strides : mêmes conventions que dans ton ancien ConvEncoderDecoder
    - la shape interne (c_o, h_o, w_o) est déterminée une fois pour toutes
      via un dummy forward à l'initialisation.
    """

    def __init__(
        self,
        in_channels: int,
        H: int,
        W: int,
        out_planes: List[int],
        strides: List[int],
        eps: float = 1e-2,
        min_channels_per_group: int = 4,
        max_groups: int = 32,
    ):
        super().__init__()
        assert len(out_planes) == len(strides), (
            "out_planes et strides doivent avoir la même longueur"
        )

        self.flatten = nn.Flatten()
        self.act = nn.ReLU(inplace=True)
        self.eps = eps
        self.min_channels_per_group = min_channels_per_group
        self.max_groups = max_groups

        self.encoder_convs = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()

        # ---- Helpers ----
        def make_gn(num_channels: int) -> nn.GroupNorm:
            C = num_channels
            num_groups = min(self.max_groups, max(1, C // self.min_channels_per_group))
            while num_groups > 1 and (C % num_groups) != 0:
                num_groups -= 1
            return nn.GroupNorm(
                num_groups=num_groups,
                num_channels=C,
                eps=self.eps,
                affine=True,
            )

        def make_norm(num_channels: int) -> nn.Module:
            return make_gn(num_channels)

        # ---- Encoder ----
        # 1ère couche
        self.encoder_convs.append(
            nn.Conv2d(
                in_channels,
                out_planes[0],
                kernel_size=3,
                stride=strides[0],
                padding=1,
                bias=False,
            )
        )
        self.encoder_norms.append(make_norm(out_planes[0]))

        # Couches suivantes
        for i in range(len(out_planes) - 1):
            self.encoder_convs.append(
                nn.Conv2d(
                    out_planes[i],
                    out_planes[i + 1],
                    kernel_size=3,
                    stride=strides[i + 1],
                    padding=1,
                    bias=False,
                )
            )
            self.encoder_norms.append(make_norm(out_planes[i + 1]))

        # ---- Détermination statique de (c_o, h_o, w_o) ----
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            out = dummy
            for conv, norm in zip(self.encoder_convs, self.encoder_norms):
                out = self.act(norm(conv(out)))
            _, self.c_o, self.h_o, self.w_o = out.shape
            self.latent_dim = self.c_o * self.h_o * self.w_o

    @property
    def feature_shape(self) -> Tuple[int, int, int]:
        """Retourne (c_o, h_o, w_o)."""
        return self.c_o, self.h_o, self.w_o

    def encode(self, x: Tensor) -> Tensor:
        """
        x : (B, C, H, W)
        return : (B, D_latent)
        """
        out = x
        for conv, norm in zip(self.encoder_convs, self.encoder_norms):
            out = self.act(norm(conv(out)))
        return self.flatten(out)


class ConvDecoder(nn.Module):
    """
    Décodeur convolutionnel :
    (B, D_latent) -> (B, out_channels, H, W)

    - feature_shape = (c_o, h_o, w_o) : doit correspondre à la sortie de ConvEncoder
    - out_planes, strides doivent correspondre à ceux utilisés dans l'encodeur
      (ils seront renversés pour construire le décodeur).
    """

    def __init__(
        self,
        out_channels: int,
        feature_shape: Tuple[int, int, int],
        out_planes: List[int],
        strides: List[int],
        eps: float = 1e-2,
        min_channels_per_group: int = 4,
        max_groups: int = 32,
    ):
        super().__init__()
        assert len(out_planes) == len(strides), (
            "out_planes et strides doivent avoir la même longueur"
        )

        self.act = nn.ReLU(inplace=True)
        self.eps = eps
        self.min_channels_per_group = min_channels_per_group
        self.max_groups = max_groups

        self.c_o, self.h_o, self.w_o = feature_shape

        self.decoder_convs = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()

        # ---- Helpers ----
        def make_gn(num_channels: int) -> nn.GroupNorm:
            C = num_channels
            num_groups = min(self.max_groups, max(1, C // self.min_channels_per_group))
            while num_groups > 1 and (C % num_groups) != 0:
                num_groups -= 1
            return nn.GroupNorm(
                num_groups=num_groups,
                num_channels=C,
                eps=self.eps,
                affine=True,
            )

        def make_norm(num_channels: int) -> nn.Module:
            return make_gn(num_channels)

        # ---- Decoder ----
        decoder_planes = list(reversed(out_planes))
        decoder_strides = list(reversed(strides))

        for i in range(len(decoder_planes) - 1):
            self.decoder_convs.append(
                nn.ConvTranspose2d(
                    decoder_planes[i],
                    decoder_planes[i + 1],
                    kernel_size=2,
                    stride=decoder_strides[i],
                    bias=False,
                )
            )
            self.decoder_norms.append(make_norm(decoder_planes[i + 1]))

        # Dernière couche de reconstruction: pas de normalisation, pas d'activation
        self.decoder_convs.append(
            nn.ConvTranspose2d(
                decoder_planes[-1],
                out_channels,
                kernel_size=2,
                stride=decoder_strides[-1],
                bias=True,
            )
        )
        self.decoder_norms.append(nn.Identity())

    def decode(self, x: Tensor) -> Tensor:
        """
        x : (B, D_latent) avec D_latent = c_o * h_o * w_o
        return : (B, out_channels, H, W)
        """
        out = x.view(-1, self.c_o, self.h_o, self.w_o)
        for conv, norm in zip(self.decoder_convs[:-1], self.decoder_norms[:-1]):
            out = self.act(norm(conv(out)))
        return self.decoder_convs[-1](out)
