"""
VGG base creator
"""
from tensorflow import keras


# F was customized for IIC
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 'M', 128, 'M', 256, 'M', 512],
    'G': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'A'],
}


class VGG:
    def __init__(self, cfg, input_shape=(28, 28, 3)):
        """
        @brief VGG network model creator to be used as base CNN for
                feature extraction
        @param cfg (dict): Summarizes the network configuration
        @param input_shape (list): Input image dimensions
        """
        self._cfg = cfg
        self.input_shape = input_shape
        self._model = None
        self.build_model()

    def build_model(self):
        """
        @brief  Model builder uses a helper function
                make_layers to read the config dict and
                create a VGG network model
        """
        inputs = keras.layers.Input(shape=self.input_shape, name='x')
        x = VGG.make_layers(self._cfg, inputs)
        self._model = keras.models.Model(inputs, x, name='VGG')

    @property
    def model(self):
        return self._model

    @staticmethod
    def make_layers(cfg,
                    inputs,
                    batch_norm: bool = True):
        """
        @brief Helper function to ease the creation of VGG
                network model
        @param cfg (dict): Summarizes the network layer 
                configuration
        @param inputs: Input from previous layer
        @param batch_norm (Bool): Whether to use batch norm
                between Conv2D and ReLU
        """
        x = inputs
        for layer in cfg:
            if layer == 'M':
                x = keras.layers.MaxPooling2D()(x)
            elif layer == 'A':
                x = keras.layers.AveragePooling2D(pool_size=3)(x)
            else:
                x = keras.layers.Conv2D(layer,
                                        kernel_size=3,
                                        padding='same',
                                        kernel_initializer='he_normal'
                                        )(x)
                if batch_norm:
                    x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Activation('relu')(x)

        return x
