"""
Python module implementing sequence-embedding layers for
Transformer-like architectures in Keras
"""

from datetime import datetime
import numpy as np

import keras
from keras import layers, ops
from keras import KerasTensor as Tensor

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]


@keras.saving.register_keras_serializable(package="Embedding")
class PositionalEmbedding(layers.Layer):
    """
    Keras implementation of the Canonical positional encoding.
    """

    def __init__(
        self,
        input_dim: PositiveInt,
        output_dim: PositiveInt,
        **kwargs
    ):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        reg_dim = output_dim + 1 if output_dim % 2 else output_dim
        aux_k = ops.arange(input_dim, dtype=keras.config.floatx())[:, None]
        aux_i = ops.arange(reg_dim // 2, dtype=keras.config.floatx())[None, :]
        self.pos_matrix = ops.concatenate((
            ops.sin(aux_k / ops.power(input_dim, 2. * aux_i / reg_dim)),
            ops.cos(aux_k / ops.power(input_dim, 2. * aux_i / reg_dim))
        ), axis=1)[None, ...]

    # pylint: disable=useless-parent-delegation
    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs: Tensor):
        return ops.take_along_axis(
            self.pos_matrix,
            ops.clip(
                inputs,
                0,
                self.input_dim - 1
            )[..., None],
            axis=-2
        )[..., :self.output_dim]


@keras.saving.register_keras_serializable(package="Embedding")
class TemporalEmbedding(layers.Layer):
    """
    Keras implementation of the Temporal embedding layer for Autoformers.
    """

    def __init__(
        self, d_model: PositiveInt,
        embed_type: str = "temporal",
        freq: str = "h",
        input_dim: int = 10000,
        **kwargs
    ):
        super(TemporalEmbedding, self).__init__(**kwargs)

        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq

        # Create the embedding layers
        if self.embed_type == "temporal":
            # Generalized Embedding (All zero-based)
            self.hour_embed = layers.Embedding(24, d_model)
            self.weekday_embed = layers.Embedding(7, d_model)
            self.day_embed = layers.Embedding(31, d_model)
            self.month_embed = layers.Embedding(12, d_model)

            if self.freq == "m":
                self.minute_embed = layers.Embedding(60, d_model)
            else:
                self.minute_embed = None

            self.time_embed = None
        else:
            self.minute_embed = None
            self.hour_embed = None
            self.weekday_embed = None
            self.day_embed = None
            self.month_embed = None

            if self.embed_type == "positional":
                self.time_embed = PositionalEmbedding(input_dim, d_model)
            elif self.embed_type == "fixed":
                self.time_embed = layers.Embedding(input_dim, d_model)
            else:
                self.time_embed = None

    def build(self, input_shape):
        if self.embed_type == "positional":
            self.time_embed.build(input_shape)

    def call(self, inputs: Tensor):
        if self.embed_type is None:
            return 0
        if self.embed_type != "temporal":
            return self.time_embed(inputs)

        month_embedded = self.month_embed(inputs[..., 0])
        day_embedded = self.day_embed(inputs[..., 1])
        weekday_embedded = self.weekday_embed(inputs[..., 2])
        hour_embedded = self.hour_embed(inputs[..., 3])
        minute_embedded = self.minute_embed(inputs[..., 4]) \
            if self.freq == "m" \
            else ops.zeros_like(hour_embedded)
        return hour_embedded + weekday_embedded + day_embedded \
            + month_embedded + minute_embedded

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "d_model": self.d_model,
                "embed_type": self.embed_type,
                "freq": self.freq,
                "minute_embed": self.minute_embed,
                "hour_embed": self.hour_embed,
                "weekday_embed": self.weekday_embed,
                "day_embed": self.day_embed,
                "month_embed": self.month_embed
            }
        )

        return config


@keras.saving.register_keras_serializable(package="Embedding")
class DataEmbedding(layers.Layer):
    """
    Keras implementation of the Positionless Data embedding layer for
    Autoformers.
    """

    def __init__(
        self,
        d_model: PositiveInt,
        dropout_rate: UnitFloat,
        embed_type: str = "temporal",
        freq: str = "h",
        **kwargs
    ):
        super(DataEmbedding, self).__init__(**kwargs)

        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.embed_type = embed_type
        self.freq = freq

        # Create the embedding layers
        self.value_embedding = layers.Conv1D(
            filters=d_model,
            kernel_size=3,
            padding="same",
            kernel_initializer="he_normal",
            activation="leaky_relu",
            use_bias=False
        )
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model,
            embed_type=embed_type,
            freq=freq
        )
        self.dropout = layers.Dropout(rate=dropout_rate)

    def build(self, input_shape):
        self.value_embedding.build(input_shape[0])
        self.temporal_embedding.build(input_shape[1])

    def call(self, inputs: tuple[Tensor, Tensor]):
        x, xm = inputs

        # Apply the value embedding to the input tensor
        x_embedded = self.value_embedding(x)

        # Apply the temporal embedding to the input tensor
        xm_embedded = self.temporal_embedding(xm)

        # Sum the two embeddings
        x_embedded += xm_embedded

        # Apply the dropout layer
        x_embedded = self.dropout(x_embedded)

        return x_embedded

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "d_model": self.d_model,
                "dropout_rate": self.dropout_rate,
                "embed_type": self.embed_type,
                "freq": self.freq
            }
        )

        return config


def timestamps_to_marks(
    timestamps: list[str],
    dt_fmt: str = "%Y-%m-%d %H:%M:%S"
) -> np.ndarray:
    """
    Function to encode a list of timestamp strings into a matrix of integers
    describing the timestamp.
    """
    def timestamp2tuple(dt_str: str):
        current_ts = datetime.strptime(dt_str, dt_fmt)
        return (current_ts.month - 1,
                current_ts.day - 1,
                current_ts.weekday(),
                current_ts.hour,
                current_ts.minute)

    marks = [timestamp2tuple(x) for x in timestamps]

    return np.array(marks)


def restore_custom_objects():
    keras.saving.get_custom_objects().update(
        {
            "Embedding>PositionalEmbedding": PositionalEmbedding,
            "Embedding>TemporalEmbedding": TemporalEmbedding,
            "Embedding>DataEmbedding": DataEmbedding
        }
    )
