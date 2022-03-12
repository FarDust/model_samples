from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from unet_autoencode.model.factory import UNetFactory
from unet_autoencode.model.blocks import Conv2DConfig


def create_model():
    model_builder = UNetFactory(
        input_shape=(256, 256, 3),
        n_filters=16,
        config=Conv2DConfig(
            kernel_size=(3, 3),
            pooling_size=(2, 2),
            padding="same",
            activation_function="relu",
            dropout=0.2,
            depth=3,
        ),
    )
    model = model_builder.build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def create_generator():
    generator = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        rotation_range=360,
        width_shift_range=1.0,
        height_shift_range=1.0,
        shear_range=30,
        zoom_range=10.0,
        fill_mode="nearest",
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
    )
    generator.fit(x_train)


def main():
    # Create the U-Net model.
    model = create_model()
    model.save("model.h5")
    model.save_weights("model_weights.h5")


if __name__ == "__main__":
    main()
