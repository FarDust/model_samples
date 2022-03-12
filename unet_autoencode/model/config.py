from dataclasses import dataclass
from typing import Union, Tuple
from unet_autoencode.settings import (
    DEFAULT_CONV_DEPTH,
    DEFAULT_CONV_ACTIVATION_FUNCTION,
    DEFAULT_CONV_PADDING,
)


@dataclass
class Conv2DConfig(object):
    """
    Configuration for a convolutional layer.
    """

    kernel_size: Union[Tuple[int, int], int] = (3, 3)
    pooling_size: Union[Tuple[int, int], int] = (2, 2)
    padding: str = DEFAULT_CONV_PADDING
    activation_function: str = DEFAULT_CONV_ACTIVATION_FUNCTION
    depth: int = DEFAULT_CONV_DEPTH
    dropout: float = 0.1
