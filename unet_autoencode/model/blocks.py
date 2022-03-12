from abc import ABCMeta
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Dropout,
    BatchNormalization,
    Activation,
    Conv2DTranspose,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from dataclasses import dataclass
from typing import Union, Tuple
from unet_autoencode.model.config import Conv2DConfig


class UNetBlock(object):
    """
    A U-Net block.
    """

    def __init__(self, n_filters, config: Conv2DConfig, batch_norm: bool):
        """
        Create a U-Net block.
        """
        self.n_filters = n_filters
        self.batch_norm = batch_norm
        self.config = config

    def unet_conv2d_layer(self, node):
        """
        Create a convolutional layer.
        """
        node = Conv2D(
            filters=self.n_filters,
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
        )(node)
        if self.batch_norm:
            node = BatchNormalization()(node)
        node = Activation(self.config.activation_function)(node)
        return node

    def conv2d_block(self, node):
        """
        Create a convolutional block.
        """
        for _ in range(self.config.depth):
            node = self.unet_conv2d_layer(node)
        return node

    def __call__(self, node):
        return self.conv2d_block(node)


class Conv2DBlock(UNetBlock):
    """
    Create a convolutional block.
    """

    def __init__(self, n_filters, config: Conv2DConfig, batch_norm=True):
        super().__init__(n_filters, config, batch_norm)

    def __call__(self, input_tensor):
        skip_node = self.conv2d_block(input_tensor)

        node = MaxPooling2D(self.config.pooling_size)(skip_node)
        node = Dropout(self.config.dropout)(node)

        return node, skip_node


class Conv2DTransposeBlock(UNetBlock):
    """
    Transpose convolutional block.
    """

    def __init__(self, n_filters, config: Conv2DConfig, batch_norm=True):
        super().__init__(n_filters, config, batch_norm)

    def __call__(self, input_tensor, skip_node):
        up_sampling_node = Conv2DTranspose(
            filters=self.n_filters,
            kernel_size=self.config.kernel_size,
            strides=self.config.pooling_size,
            padding=self.config.padding,
        )(input_tensor)

        up_sampling_node = concatenate([up_sampling_node, skip_node])
        node = Dropout(self.config.dropout)(up_sampling_node)

        node = self.conv2d_block(node)

        return node
