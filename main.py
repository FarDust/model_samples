import datetime
from tensorflow.data import Dataset
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


def create_train_generator(directory: str, input_shape: tuple, batch_size: int):

    generator_kargs = {
        "rescale": 1./255,
        "rotation_range": 360,
        "width_shift_range": 1.0,
        "height_shift_range": 1.0,
        "shear_range": 30,
        "zoom_range": 10.0,
        "fill_mode": "nearest",
        "horizontal_flip": True,
        "vertical_flip": True,
        "validation_split": 0.2,
    }

    train_images_datagen = ImageDataGenerator(
        **generator_kargs
    )

    train_masks_datagen = ImageDataGenerator(
        **generator_kargs
    )

    seed = int(datetime.now())

    shared_kargs_params = {
        "seed": seed,
        "target_size": input_shape,
        "batch_size": batch_size,
        "class_mode": None,
        "interpolation":"bilinear",
    }

    return zip(train_images_datagen.flow_from_directory(
        directory,
        subset="images",
        **shared_kargs_params
    ), train_masks_datagen.flow_from_directory(
        directory,
        subset="masks",
        **shared_kargs_params
    ))
    


def main():
    # Create the U-Net model.
    model = create_model()
    train_generator = create_train_generator('data/train', (256, 256, 3), batch_size=32)
    model.fit(train_generator, steps_per_epoch=2000, epochs=50)
    model.save("model.h5")
    model.save_weights("model_weights.h5")


if __name__ == "__main__":
    main()
