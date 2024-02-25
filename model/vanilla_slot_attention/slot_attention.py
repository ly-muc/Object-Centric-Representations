# Custom Implementation based on the paper
# "Slot Attention" (https://arxiv.org/abs/2006.15055)

import numpy as np
import torch

from typing import Optional, Tuple
from torch.autograd import Variable
from torch import nn, Tensor
from unet import UNet


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def build_grid(resolution: Tuple[int, int]):
    """Function creates grid for position embedding
    Args:
        resolution (tuple[int, int]): Image resolution

    Returns:
        numpy.array: Grid
    """
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)

    return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(nn.Module):

    def __init__(self, hidden_size: int, resolution: Tuple[int, int]):
        """Builds the soft position embedding layer.
        Args:
            hidden_size (int): Size of input feature dimension.
            resolution (tuple[int, int]): Tuple of integers specifying width and height of grid.
        """
        super(SoftPositionEmbed, self).__init__()
        self.resolution = resolution
        self.dense = nn.Linear(4, hidden_size, bias=True)
        self.grid = torch.Tensor(build_grid(
            resolution)).view((-1, 4)).unsqueeze(0)

    def forward(self, inputs):

        # spatial flatten [batch_size, channels, width, height] -> [batch_size, width * height, channels]
        inputs = inputs.view(
            *inputs.shape[:2], inputs.shape[-1]*inputs.shape[-2]).permute(0, 2, 1)
        embedding = self.dense(self.grid.to(inputs.device))

        return inputs + embedding


def unstack_and_split(input, batch_size):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = input.view(
        (batch_size, -1, * input.shape[1:]))  # shape: [batch_size, num_slots, channels, width, height]

    mask = unstacked[:, :, 3, :, :].unsqueeze(2)
    channels = unstacked[:, :, :3, :, :]

    return channels, mask


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""

    # [batch_size*num_slots, slot_dim, 1, 1]
    slots = slots.view(-1, slots.shape[-1]).unsqueeze(-1).unsqueeze(-1)
    # shape [batch_size*num_slots, slot_dim, height, width]
    return slots.tile((1, 1, *resolution))


class SlotAttention(nn.Module):
    """This class implements slot attention for object discoverys

    Attributes:
        num_slots: Number of slots defined at test time (can be changed for inference)
    """

    def __init__(self, num_slots: int, num_iterations: int, input_dim: int, slot_dim: int, mlp_dim: int,
                 device="cuda", implicit_diff=False, eps=1e-6):

        super(SlotAttention, self).__init__()

        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.device = device
        self.eps = eps
        self.implicit_diff = implicit_diff

        self.k_linear = nn.Linear(input_dim, slot_dim, bias=False)
        self.q_linear = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v_linear = nn.Linear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.input_layer_norm = nn.LayerNorm(input_dim)
        self.slots_layer_norm = nn.LayerNorm(slot_dim)
        self.mlp_layer_norm = nn.LayerNorm(input_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.slots_mu = nn.Parameter(
            torch.randn(1, 1, slot_dim)).to(self.device)
        self.slots_logsigma = nn.Parameter(
            torch.zeros(1, 1, slot_dim)).to(self.device)
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, slot_dim))

    def step(self, slots, k, v, batch_size):
        slots_prev = slots
        slots = self.slots_layer_norm(slots)

        # Attention
        q = self.q_linear(slots)

        # shape [batch_size, input_dim, slot_dim]
        dots = torch.einsum('bid,bjd->bij', q, k)
        # shape [batch_size, input_dim, slot_dim]
        attn = dots.softmax(dim=1) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # shape [batch_size, input_dim, slot_dim]
        updates = torch.einsum('bjd,bij->bid', v, attn)

        slots = self.gru(
            updates.contiguous().view(-1, self.slot_dim),
            slots_prev.contiguous().view(-1, self.slot_dim)
        )

        slots = slots.reshape(batch_size, -1, self.slot_dim)
        slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn

    def forward(self, inputs, slot_init: Optional[Tensor] = None):

        # get input size
        batch_size, _, _ = inputs.shape

        # apply layer norm to input
        inputs = self.input_layer_norm(inputs)

        # intial slot representation
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.num_slots, -1)

        # slots shape [batch_size, num_slots, num_inputs]
        slots = mu + sigma * torch.randn(mu.shape, device=self.device)

        if slot_init is not None:
            slots = slot_init

        # apply linear transformations
        # shape [batch_size, height*width, input_dim]
        k = self.k_linear(inputs) * self.slot_dim ** -.5
        # shape [batch_size, height*width, input_dim]
        v = self.v_linear(inputs)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots, attn = self.step(slots, k, v, batch_size)

        # object representation as fixed point paper
        if self.implicit_diff:
            slots, attn = self.step(slots.detach(), k, v, batch_size)

        return slots, attn


class SlotAutoEncoder(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""

    def __init__(
        self,
        device,
        resolution: Tuple[int, int] = (128, 128),
        num_slots=15,
        slot_dim=64,
        num_iterations=5,
        vae=False,
        implicit_diff=False,
        encoder: Optional[nn.Module] = None
    ):
        """
        Args:
            device: Cpu or Cuda
            resolution (tuple[int, int], optional): Resolution of the
                input images. Defaults to (64, 64).
            num_slots (int, optional): Number of individual slots in Slot
                Attention. Defaults to 10.
            slot_dim (int, optional): Size of one Slot. Defaults to 64.
            num_iterations (int, optional): Number of iterations f.
                Defaults to 5.
            implicit_diff (bool): Use implicit differentiation
        """

        super(SlotAutoEncoder, self).__init__()

        self.resolution = resolution
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.vae = vae

        self.enc_channels = 64

        if encoder == "unet":
            self.encoder_cnn = UNet(num_blocks=6, in_chnls=3, out_chnls=64)
        else:
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64,
                          kernel_size=5, padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=5, padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=5, padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=self.enc_channels,
                          kernel_size=5, padding='same'),
                nn.ReLU(),
            )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels=slot_dim, out_channels=64,
                               kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=4, kernel_size=3,
                               stride=1))

        self.decoder_init = (8, 8)

        # resolution = output shape of encoder
        self.encoder_pos = SoftPositionEmbed(
            hidden_size=self.enc_channels, resolution=resolution)
        self.decoder_pos = SoftPositionEmbed(
            hidden_size=self.slot_dim, resolution=self.decoder_init)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.enc_channels, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2 *
                      self.enc_channels if self.vae else self.enc_channels)
        )

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            num_iterations=self.num_iterations,
            input_dim=self.enc_channels,
            slot_dim=self.slot_dim,
            mlp_dim=128,
            device=device,
            implicit_diff=implicit_diff)

        self.input_layer_norm = nn.LayerNorm(self.enc_channels)

    def forward(self, input, slot_init=None):

        batch_size, _, _, _ = input.shape

        # CNN backbone
        # shape: [batch_size, channels, width, height]
        cnn_features = self.encoder_cnn(input)

        # add positional encoding
        # shape: [batch_size, width * height, input_dim]
        z = self.encoder_pos(cnn_features)

        # MLP head
        # shape: [batch_size, width * height, input_dim]
        features = self.mlp(self.input_layer_norm(z))

        if self.vae:
            mu = features[..., :self.enc_channels]
            logvar = features[..., self.enc_channels:]
            features = reparametrize(mu, logvar)

        # Slot Attention
        if slot_init is not None:
            slots, attn = self.slot_attention(features, slot_init)
        else:
            # shape: [batch_size, num_slots, slot_dim]
            slots, attn = self.slot_attention(features)

        # shape: [batch_size*num_slots, slot_dim, height_init, width_init]
        z = spatial_broadcast(slots, self.decoder_init)
        z = self.decoder_pos(z)

        # shape: [batch_size * num_slots, slot_dim, width_init * height_init]
        z = z.permute(0, 2, 1)
        # shape: [batch_size * num_slots, slot_dim, width_init, height_init]
        z = z.view(*z.shape[:2], *self.decoder_init)

        # CNN Decoder
        z = self.decoder_cnn(z)

        # Reconstructions, Alpha Mask
        recons, masks = unstack_and_split(z, batch_size=batch_size)

        recon_combined = torch.sum(masks.softmax(dim=1) * recons, dim=1)

        if self.vae:
            return recon_combined, recons, masks, slots, mu, logvar, attn
        else:
            return recon_combined, recons, masks, slots, cnn_features

    def encode(self, input):
        return self.encoder_cnn(input)

    def decode(self, input):
        return self.decoder_cnn(input)

    def encode_and_cluster(self, input):

        batch_size, _, _, _ = input.shape

        # CNN backbone
        # shape: [batch_size, channels, width, height]
        z = self.encoder_cnn(input)

        # add positional encoding
        # shape: [batch_size, width * height, input_dim]
        z = self.encoder_pos(z)

        # MLP head
        # shape: [batch_size, width * height, input_dim]
        features = self.mlp(self.input_layer_norm(z))

        if self.vae:
            mu = features[..., :self.enc_channels]
            logvar = features[..., self.enc_channels:]
            features = reparametrize(mu, logvar)

        # shape: [batch_size, num_slots, slot_dim]
        slots, attn = self.slot_attention(features)

        return slots, attn
