import torch
from torch import Tensor
import matplotlib.pyplot as plt


def concatenate_images(
    image: Tensor,
    segmentation: Tensor,
    reconstruction: Tensor,
    attention_mask: Tensor,
    slot_recons: Tensor,
    N: int = 8
) -> Tensor:
    """Function for concatenating images for visualization.

    Args:
        image (Tensor): Input image. Shape:
            [batch_size, channels, height, width]
        segmentation (Tensor): Ground truth. Shape:
            [batch_size, channels, heigh, width]
        reconstruction (Tensor): Decoder reconstruction. Shape:
            [batch_size, channels, heigh, width]
        attention_mask (Tensor): Shape: [batch_size, channels, heigh, width]
        slot_recons (Tensor): Per slot reconstructions. Shape:
            [batch_size, num_slots, channels, heigh, width]
        N (int, optional): Max number per batch. Defaults to 8.

    Returns:
        Tensor: Concatenated images
    """
    _, _, H, W = image.shape

    # resize inputs

    # shape: [batch_size, 1, channels, height, width]
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    # shape: [batch_size, 1, channels, height, width]
    segmentation = segmentation[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    # shape: [batch_size, 1, channels, height, width]
    reconstruction = reconstruction[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    # shape: [batch_size, 1, channels, height, width]
    attention_mask = attention_mask[:8].unsqueeze(1)
    # shape: [batch_size, 1, channels, height, width]
    slot_recons = slot_recons[:N].expand(-1, -1, 3, H, W)

    # shape: [batch_size, 4+num_slots, channels, height, width]
    return torch.cat((image, reconstruction, segmentation,
                      attention_mask, slot_recons), dim=1).view(-1, 3, H, W)


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


class ColorMap():

    def __init__(self, num_objects: int, name='hsv'):

        self.num_colors = num_objects
        self.cmap = get_cmap(self.num_colors + 1, name=name)
        self.r_get = {i: self.cmap(i)[0] for i in range(self.num_colors)}.get
        self.g_get = {i: self.cmap(i)[1] for i in range(self.num_colors)}.get
        self.b_get = {i: self.cmap(i)[2] for i in range(self.num_colors)}.get

    def __call__(self, inputs: Tensor) -> Tensor:
        """Function returns rgb image from uint8 segmentation input.

        Args:
            inputs (Tensor): Input segmenation. Shape:
                [batch_size, 1, height, width]

        Returns:
            Tensor: [batch_size, 3, height, width]
        """

        is_cuda = inputs.is_cuda

        inputs = inputs.repeat(1, 3, 1, 1).float()

        if is_cuda:
            inputs = inputs.to("cpu")

        inputs[:, 0, :, :].apply_(self.r_get)
        inputs[:, 1, :, :].apply_(self.g_get)
        inputs[:, 2, :, :].apply_(self.b_get)

        if is_cuda:
            inputs = inputs.to("cuda")

        return inputs
