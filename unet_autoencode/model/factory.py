from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras import Model
from unet_autoencode.model.blocks import (
    Conv2DBlock,
    Conv2DTransposeBlock,
    UNetBlock,
    Conv2DConfig,
)
import numpy as np


class UNetFactory:
    def __init__(self, input_shape, n_filters, config: Conv2DConfig, batch_norm=True):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.config = config
        self.batch_norm = batch_norm

        self.latent_space_filter_rate = 16
        block_qty = int(np.log2(self.latent_space_filter_rate))

        self.__encoder_blocks = [
            Conv2DBlock(self.n_filters * (2 ** i), self.config, self.batch_norm)
            for i in range(block_qty)
        ]

        self.__decorder_blocks = [
            Conv2DTransposeBlock(
                self.n_filters * (2 ** i), self.config, self.batch_norm
            )
            for i in range(block_qty)
        ]
        self.__decorder_blocks.reverse()
        self.__skip_nodes = list()

    def build_input(self):
        """
        Create the input layer.
        """
        return Input(self.input_shape)

    def build_encoder(self, input_layer):
        """
        Build the encoder.
        """
        for block in self.__encoder_blocks:
            input_layer, skip_node = block(input_layer)
            self.__skip_nodes.append(skip_node)
        return input_layer

    def build_latent(self, input_layer):
        """
        Build the latent space.
        """
        return UNetBlock(
            self.n_filters * self.latent_space_filter_rate, self.config, self.batch_norm
        )(input_layer)

    def build_decoder(self, input_layer):
        """
        Build the decoder.
        """
        node = input_layer
        for block in self.__decorder_blocks:
            node = block(node, self.__skip_nodes.pop())
        return node

    def build_output(self, input_layer):
        """
        Build the output layer.
        """
        outputs = Conv2D(1, (1, 1), activation="sigmoid")(input_layer)
        return outputs

    def build_model(self):
        """
        Create a U-Net model.
        """
        input_layer = self.build_input()
        node = self.build_encoder(input_layer)
        node = self.build_latent(node)
        node = self.build_decoder(node)
        node = self.build_output(node)
        return Model(inputs=[input_layer], outputs=[node])
