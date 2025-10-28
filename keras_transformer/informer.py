"""
Python module implementing the necessary layers for an
Informer in Keras
"""

import math

import keras
from keras import layers, ops
from keras import KerasTensor as Tensor

from .embed import DataEmbedding

from pydantic import Field, PositiveInt
from typing import Annotated, Optional

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]


@keras.saving.register_keras_serializable(package="Informer")
class ProbAttentionLayer(layers.Layer):
    """
    Keras implementation of the autocorrelation attention
    mechanism
    """

    def __init__(self, params: dict, **kwargs):
        super(ProbAttentionLayer, self).__init__(**kwargs)
        self.params = params

        self.k = params.get("k", 1)
        self.h = params.get("h", 2)

        self.d_keys: PositiveInt = params.get("d_keys", None)
        self.d_values: PositiveInt = params.get("d_values", None)

        self.dropout = layers.Dropout(params.get("dropout_rate", 1e-3))

        self.output_attention = params.get("output_attention", False)

        self.mask_flag = params.get("mask_flag", True)

        self.mix = params.get("mix", False)

        self.seed = keras.random.SeedGenerator()

        self.query_proj: layers.Dense = None
        self.key_proj: layers.Dense = None
        self.value_proj: layers.Dense = None
        self.out_proj: layers.Dense = None

    def build(self, input_shape: tuple[tuple[int, ...], ...]):
        self.d_keys = self.d_keys or input_shape[0][-1] // self.h
        self.d_values = self.d_values or input_shape[0][-1] // self.h

        self.query_proj = layers.Dense(self.h * self.d_keys)
        self.key_proj = layers.Dense(self.h * self.d_keys)
        self.value_proj = layers.Dense(self.h * self.d_values)
        self.out_proj = layers.Dense(input_shape[0][-1])

    def _prob_qk(
        self,
        queries: Tensor,
        keys: Tensor,
        sample_k: int,
        n_top: int
    ) -> tuple[Tensor, Tensor]:
        _, _, q_len, _ = queries.shape
        _, _, k_len, _ = keys.shape

        # Calculate sampled Q_k
        k_expand = ops.tile(
            keys[:, :, None, :, :],
            (1, 1, q_len, 1, 1)
        )
        index_sample = keras.random.randint(
            (q_len, sample_k),
            0, k_len,
            seed=self.seed
        )
        k_sample = k_expand[:, :, ops.arange(q_len)[:, None], index_sample, :]
        q_k_sample = ops.matmul(
            queries[..., None, :],
            ops.transpose(k_sample, (0, 1, 2, 4, 3))
        )[..., 0, :]

        # Find the top_k query with sparsity measurement
        m = ops.max(q_k_sample, -1) - ops.divide(ops.sum(q_k_sample, -1), k_len)
        m_top = ops.top_k(m, n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        q_reduce = ops.take_along_axis(queries, m_top[..., None], -2)
        return ops.matmul(q_reduce, ops.transpose(keys, (0, 1, 3, 2))), m_top

    def _prob_mask(self, length: int, index: Tensor, scores: Tensor):
        mask = ops.triu(
            ops.ones(
                (length, scores.shape[-1]),
                dtype="bool"
            ), 1
        )[None, None, ...]
        indicator = ops.take_along_axis(mask, index[..., None], -2)
        return indicator

    def call(self, inputs: tuple[Tensor, Tensor, Tensor]
             ) -> tuple[Tensor, Tensor]:
        queries, keys, values = inputs

        _, q_len, d_out = queries.shape
        _, k_len, _ = keys.shape
        _, v_len, _ = values.shape
        h = self.h

        queries = ops.reshape(
            self.query_proj(queries),
            (-1, q_len, h, self.d_keys)
        )
        keys = ops.reshape(
            self.key_proj(keys),
            (-1, k_len, h, self.d_keys)
        )
        values = ops.reshape(
            self.value_proj(values),
            (-1, k_len, h, self.d_values)
        )

        queries = ops.transpose(queries, (0, 2, 1, 3))
        keys = ops.transpose(keys, (0, 2, 1, 3))
        values = ops.transpose(values, (0, 2, 1, 3))

        u_part = self.k * int(math.ceil(math.log(k_len)))
        u = self.k * int(math.ceil(math.log(q_len)))

        u_part = u_part if u_part < k_len else k_len
        u = u if u < q_len else q_len

        scores_top, index = self._prob_qk(
            queries=queries,
            keys=keys,
            sample_k=u_part,
            n_top=u
        )

        # Normalization
        scores_top /= math.sqrt(d_out)

        # Context generation + update
        if not self.mask_flag:
            v_sum = ops.mean(values, -2, keepdims=True)
            context = ops.tile(v_sum, (1, 1, q_len, 1))
        else:
            assert q_len == values.shape[-2]
            context = ops.cumsum(values, -2)
            attn_mask = self._prob_mask(q_len, index, scores_top)
            scores_top = ops.where(attn_mask, -math.inf, scores_top)
        attn = ops.softmax(scores_top, -1)

        bs = ops.shape(context)[0]
        attn_values = ops.cast(ops.matmul(attn, values), dtype=attn.dtype)

        update_indices = ops.reshape(
            ops.stack(
                [
                    ops.broadcast_to(
                        ops.arange(bs)[
                            :, None, None], index.shape),
                    ops.broadcast_to(ops.arange(h)[None, :, None], index.shape),
                    index
                ], axis=-1
            ), (-1, 3)
        )

        context = ops.scatter_update(
            inputs=context,
            indices=update_indices,
            updates=ops.reshape(attn_values, (-1, attn_values.shape[-1]))
        )

        if not self.mix:
            context = ops.transpose(context, (0, 2, 1, 3))
        out = self.out_proj(
            ops.reshape(
                context,
                (-1, q_len, h * self.d_values)
            )
        )

        if self.output_attention:
            attn_ext = ops.ones([bs, h, v_len, v_len], dtype=attn.dtype) / v_len

            attn_ext = ops.scatter_update(
                attn_ext,
                update_indices,
                ops.reshape(attn, (-1, attn.shape[-1]))
            )

            attn_ext = ops.transpose(attn_ext, (0, 2, 1, 3))
            return self.dropout(out), attn_ext
        else:
            return self.dropout(out), None


@keras.saving.register_keras_serializable(package="Informer")
class ProbEncoderLayer(layers.Layer):
    """
    Keras implementation of a single Informer encoder layer
    """

    def __init__(self, params: dict, **kwargs):
        super(ProbEncoderLayer, self).__init__(**kwargs)
        self.params = params

        self.d_ff: int = params.get("d_ff", None)
        self.activation = params.get("activation", "gelu")

        self.attn_layer = ProbAttentionLayer({**params, "mask_flag": False})

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

    def call(self, inputs: Tensor) -> Tensor:
        x, attn = self.attn_layer([inputs] * 3)

        x = self.norm_1(inputs + self.dropout(x))

        return self.norm_2(x + self.ff_layer(x)), attn

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


@keras.saving.register_keras_serializable(package="Informer")
class ProbEncoder(layers.Layer):
    """
    Keras implementation of the Transformer encoder block
    """

    def __init__(self, params: dict, **kwargs):
        super(ProbEncoder, self).__init__(**kwargs)
        self.params = params

        self.d_model = params["d_model"]
        self.mask_flag = params.get("mask_flag", True)

        self.embed = DataEmbedding(
            params["d_model"],
            params["dropout_rate"],
            params["embed_type"],
            params.get("freq")
        )

        n_layers = params["N"]
        self.enc_layers = [
            ProbEncoderLayer(params) for _ in range(n_layers)
        ]

        self.norm_layer = layers.LayerNormalization(
            axis=-1) if params["normalize"] else layers.Identity()

    def build(self, input_shape):
        embedded_shape = [*input_shape[0][:-1], self.d_model]
        for encoder_layer in self.enc_layers:
            encoder_layer.build(embedded_shape)

    def call(
        self,
        inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Optional[Tensor], ...]]:
        attns = []
        x = self.embed(inputs)

        for enc_layer in self.enc_layers:
            x, attn = enc_layer(x)
            attns.append(attn)

        return self.norm_layer(x), attns


@keras.saving.register_keras_serializable(package="Informer")
class ProbDecoderLayer(layers.Layer):
    """
    Keras implementation of a single Informer decoder layer
    """

    def __init__(self, params: dict, **kwargs):
        super(ProbDecoderLayer, self).__init__(**kwargs)
        self.params = params

        self.d_ff: int = params.get("d_ff", None)
        self.activation = params.get("activation", "gelu")

        self.self_attn = ProbAttentionLayer({**params, "mask_flag": True})
        self.cross_attn: layers.MultiHeadAttention = None

        self.dropout = layers.Dropout(params.get("dropout_rate", 0.1))
        self.norm_1 = layers.LayerNormalization(
        ) if params["normalize"] else layers.Identity()
        self.norm_2 = layers.LayerNormalization(
        ) if params["normalize"] else layers.Identity()
        self.ff_layer: keras.Sequential = None
        self.norm_3 = layers.LayerNormalization(
        ) if params["normalize"] else layers.Identity()

    def build(self, input_shape: tuple[tuple[int, ...], ...]):
        self.d_ff = self.d_ff or 4 * input_shape[0][-1]

        h = self.params.get("h", 2)
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=h,
            key_dim=self.params.get(
                "d_keys",
                input_shape[0][-1] // h
            )
        )

        self.ff_layer = keras.Sequential([
            layers.Dense(units=self.d_ff, activation=self.activation),
            self.dropout,
            layers.Dense(units=input_shape[0][-1], activation=None),
            self.dropout
        ])

    def call(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        x, cross = inputs

        x += self.dropout(self.self_attn([x] * 3)[0])
        x = self.norm_1(x)

        x += self.dropout(self.cross_attn(x, cross, use_causal_mask=False))
        x = self.norm_2(x)

        return self.norm_3(x + self.ff_layer(x))

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


@keras.saving.register_keras_serializable(package="Informer")
class ProbDecoder(layers.Layer):
    """
    Keras implementation of the Transformer decoder block
    """

    def __init__(self, params: dict, **kwargs):
        super(ProbDecoder, self).__init__(**kwargs)
        self.params = params

        self.d_model = params["d_model"]

        self.embed = DataEmbedding(
            params["d_model"],
            params["dropout_rate"],
            params["embed_type"],
            params.get("freq")
        )

        n_layers = params["M"]
        self.dec_layers = [
            ProbDecoderLayer(params) for _ in range(n_layers)
        ]

        self.norm_layer = layers.LayerNormalization(
            axis=-1) if params["normalize"] else layers.Identity()
        self.out_proj = layers.Dense(params["d_out"])

    def build(self, input_shape):
        embedded_shape = input_shape[-1]
        for decoder_layer in self.dec_layers:
            decoder_layer.build([embedded_shape] * 2)

    def call(
        self,
        inputs: tuple[Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Optional[Tensor], ...]]:
        x, xm, cross = inputs
        x = self.embed([x, xm])

        for dec_layer in self.dec_layers:
            x = dec_layer([x, cross])

        return self.out_proj(self.norm_layer(x))


def restore_custom_objects():
    keras.saving.get_custom_objects().update({
        "Informer>ProbAttentionLayer": ProbAttentionLayer,
        "Informer>ProbEncoderLayer": ProbEncoderLayer,
        "Informer>ProbEncoder": ProbEncoder,
        "Informer>ProbDecoderLayer": ProbDecoderLayer,
        "Informer>ProbDecoder": ProbDecoder
    })
