import torch
from torch import nn, Tensor

from KSIC.models.nn.conv_encoder_decoder import ConvEncoder, ConvDecoder
from KSIC.models.nn.mlp_blocks import build_mlp

class AutoEncoder(nn.Module):
    def __init__(self, params: dict, z_dim: int, drone_dim: int):
        super().__init__()
        self.params = params
        self.z_dim = z_dim

        dim_hidden = self.params["MLP"]["dim_hidden"]
        num_hidden = self.params["MLP"]["num_hidden_layers"]
        out_planes = self.params["CNN"]["out_planes"]
        strides = self.params["CNN"]["strides"]
        activation = self.params["activation"]

        if drone_dim == 2:
            in_channels = 2
            out_channels = 1
            self.num_views = 1
        elif drone_dim == 3:
            in_channels = 4
            out_channels = 2
            self.num_views = 2
        else:
            raise ValueError(f"Invalid drone dimension {drone_dim}")

        self.encoder_cnn = ConvEncoder(
            in_channels=in_channels,
            H=128,
            W=128,
            out_planes=out_planes,
            strides=strides,
        )
        self.w_dim = self.encoder_cnn.latent_dim

        self.encoder_mlp = build_mlp(
            self.w_dim,
            dim_hidden,
            self.z_dim,
            num_hidden,
            activation
        )

        self.decoder_mlp = build_mlp(
            self.z_dim,
            dim_hidden,
            self.w_dim,
            num_hidden,
            activation
        )

        self.decoder_cnn = ConvDecoder(
            out_channels=out_channels,
            feature_shape=self.encoder_cnn.feature_shape,
            out_planes=out_planes,
            strides=strides,
        )

    def project(self, y: Tensor) -> Tensor:
        """
        Projection: image space to the observable space
        """
        device = next(self.parameters()).device
        y = y.to(device)
        w = self.encoder_cnn.encode(y)
        return self.encoder_mlp(w)

    def reconstruct(self, z: Tensor) -> Tensor:
        """
        Reconstruction: observable space to the image space
        """
        w = self.decoder_mlp(z)
        return self.decoder_cnn.decode(w)

    def forward(self, y: Tensor) -> Tensor:
        z = self.project(y)
        return self.reconstruct(z)

    def batch_projection(self, y_gt: Tensor) -> Tensor:
        b_size = y_gt.size(0)
        n_steps = y_gt.size(1)
        res = y_gt.size(-1)

        y_gt_flat = torch.reshape(y_gt, (b_size*n_steps, self.num_views*2, res, res))
        z_proj_flat = self.project(y_gt_flat)
        return torch.reshape(z_proj_flat, (b_size, n_steps, self.z_dim))

