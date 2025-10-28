"""
Module implementing an Recurrent Encoder-Decoder stacks compatible with the
Generic Transformer
"""

import keras
from keras import KerasTensor as Tensor
from keras import Layer
from keras import layers

from .embed import DataEmbedding


@keras.saving.register_keras_serializable(package="Recurrent")
class RecEncoderLayer(Layer):
    """
    Recurrent Encoder Layer
    """

    def __init__(self, params: dict, **kwargs):
        super().__init__(**kwargs)
        self.params = params

        self.d_ff: int = params.get("d_ff", None)
        self.activation = params.get("activation", "gelu")

        self.rnn_type = params.get("rnn_type", "lstm")
        self.h = params.get("h", 16)

        match self.rnn_type:
            case "lstm":
                rnn_class = layers.LSTM
            case "gru":
                rnn_class = layers.GRU
            case "simple":
                rnn_class = layers.SimpleRNN

        self.rnn = layers.Bidirectional(
            rnn_class(
                units=self.h,
                activation=None,
                return_sequences=True,
                return_state=True
            )
        )

        self.dropout = layers.Dropout(params.get("dropout_rate", 0.1))

        self.norm_1 = layers.LayerNormalization(
        ) if params["normalize"] else layers.Identity()
        self.ff_layer: keras.Sequential = None
        self.norm_2 = layers.LayerNormalization(
        ) if params["normalize"] else layers.Identity()

    def build(self, input_shape: tuple[int, ...]):
        self.d_ff = self.d_ff or 4 * input_shape[-1]

        self.ff_layer = keras.Sequential([
            layers.Dense(units=self.d_ff, activation=self.activation),
            self.dropout,
            layers.Dense(units=input_shape[-1], activation=None),
            self.dropout
        ])

    def call(
        self,
        inputs: Tensor,
        training=None
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        x, *state = self.rnn(
            inputs,
            training=training
        )
        x = self.norm_1(self.dropout(x))

        return self.norm_2(self.ff_layer(x)), state

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


@keras.saving.register_keras_serializable(package="Recurrent")
class RecEncoder(Layer):
    """
    Recurrent Encoder Stack
    """

    def __init__(self, params: dict, **kwargs):
        super().__init__(**kwargs)
        self.params = params

        self.d_model = params["d_model"]
        self.embed = DataEmbedding(
            params["d_model"],
            params["dropout_rate"],
            params["embed_type"],
            params.get("freq")
        )

        n_layers = params["N"]
        self.enc_layers = [RecEncoderLayer(params) for _ in range(n_layers)]

    def build(self, input_shape):
        embedded_shape = [*input_shape[0][:-1], self.d_model]
        for encoder_layer in self.enc_layers:
            encoder_layer.build(embedded_shape)

    def call(
        self,
        inputs: tuple[Tensor, Tensor]
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        # The role of "Attention Tensors" is assumed by the intermediate
        # sequences
        attns = []
        x = self.embed(inputs)

        for enc_layer in self.enc_layers:
            x, state = enc_layer(x)
            attns.append(x)

        return state, tuple(attns)

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


@keras.saving.register_keras_serializable(package="Recurrent")
class RecDecoderLayer(Layer):
    """
    Recurrent Decoder Layer
    """

    def __init__(self, params: dict, **kwargs):
        super().__init__(**kwargs)
        self.params = params

        self.d_ff: int = params.get("d_ff", None)
        self.activation = params.get("activation", "gelu")

        self.rnn_type = params.get("rnn_type", "lstm")
        self.h = params.get("h", 16)

        match self.rnn_type:
            case "lstm":
                rnn_class = layers.LSTM
            case "gru":
                rnn_class = layers.GRU
            case "simple":
                rnn_class = layers.SimpleRNN

        self.rnn = rnn_class(
            units=self.h,
            activation=None,
            return_sequences=True,
            return_state=False
        )

        self.dropout = layers.Dropout(params.get("dropout_rate", 0.1))
        self.norm_1 = layers.LayerNormalization(
        ) if params["normalize"] else layers.Identity()
        self.ff_layer: keras.Sequential = None
        self.norm_2 = layers.LayerNormalization(
        ) if params["normalize"] else layers.Identity()

    def build(
            self, input_shape: tuple[
                tuple[int, ...] | tuple[tuple[int, ...], ...],
                ...
            ]):
        self.d_ff = self.d_ff or 4 * input_shape[-1]

        self.ff_layer = keras.Sequential([
            layers.Dense(units=self.d_ff, activation=self.activation),
            self.dropout,
            layers.Dense(units=input_shape[0][-1], activation=None),
            self.dropout
        ])

        self.init_size = len(input_shape[1]) // 2

    def call(
        self,
        inputs: tuple[Tensor, tuple[Tensor, ...]],
        training=None
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        x, init = inputs

        x = self.rnn(
            x,
            initial_state=init[:self.init_size],
            training=training
        )
        x = self.norm_1(self.dropout(x))

        return self.norm_2(self.ff_layer(x))

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


@keras.saving.register_keras_serializable(package="Recurrent")
class RecDecoder(Layer):
    """
    Recurrent Decoder Stack
    """

    def __init__(self, params: dict, **kwargs):
        super().__init__(**kwargs)
        self.params = params

        self.d_model = params["d_model"]
        self.embed = DataEmbedding(params["d_model"],
                                   params["dropout_rate"],
                                   params["embed_type"],
                                   params.get("freq"))

        n_layers = params["M"]
        self.dec_layers = [RecDecoderLayer(params) for _ in range(n_layers)]

        self.out_proj = layers.Dense(params["d_out"])

    def build(self, input_shape):
        embedded_shape = [[*input_shape[0][:-1], self.d_model], input_shape[-1]]
        for decoder_layer in self.dec_layers:
            decoder_layer.build(embedded_shape)

    def call(
        self,
        inputs: tuple[Tensor, Tensor, Tensor | tuple[Tensor, ...]]
    ) -> Tensor:
        x = self.embed(inputs[:2])

        for dec_layer in self.dec_layers:
            x = dec_layer([x, inputs[-1]])

        return self.out_proj(x)

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


def restore_custom_objects():
    keras.saving.get_custom_objects().update(
        {
            "Recurrent>RecEncoderLayer": RecEncoderLayer,
            "Recurrent>RecEncoder": RecEncoder,
            "Recurrent>RecDecoderLayer": RecDecoderLayer,
            "Recurrent>RecDecoder": RecDecoder
        }
    )
