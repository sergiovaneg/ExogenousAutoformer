"""
Python module implementing the necessary layers for an
Autoformer in Keras
"""

import math
from itertools import zip_longest
from functools import partial

import keras
from keras import layers, ops
from keras import KerasTensor as Tensor

from scipy import signal

from .embed import DataEmbedding

from collections.abc import Sequence
from pydantic import Field, PositiveInt
from typing import Annotated, Optional


DEFAULT_PAD_MODE = "edge" if keras.backend.backend() == "jax" else "constant"
UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]


@keras.saving.register_keras_serializable(package="Autoformer")
class MADecompositionLayer(layers.Layer):
    """
    Moving-Average-based seasonal-trend decomposition
    """

    def __init__(self, tau: int, **kwargs):
        super(MADecompositionLayer, self).__init__(**kwargs)
        self.tau = tau

    def call(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        trend = ops.average_pool(
            ops.concatenate(
                [
                    ops.repeat(inputs[..., :1, :], self.tau, -2),
                    inputs,
                    ops.repeat(inputs[..., -1:, :], self.tau, -2)
                ],
                axis=-2
            ), self.tau, strides=1,
            padding="same", data_format="channels_last"
        )[..., self.tau:-self.tau, :]
        seasonality = inputs - trend

        return seasonality, trend


@keras.saving.register_keras_serializable(package="Autoformer")
class IIRDecompositionLayer(layers.Layer):
    """
    IIR-filter-based seasonal-trend decomposition (4th-order by default)
    """

    def __init__(
            self,
            tau: int,
            order: int = 2,
            filter_mode: str = "associative",
            **kwargs
    ):
        super(IIRDecompositionLayer, self).__init__(**kwargs)
        self.tau = tau
        self.filter_mode = filter_mode

        sos = ops.convert_to_tensor(
            signal.iirfilter(
                order,
                1. / self.tau,
                btype="highpass",
                output="sos",
                ftype="butter",
                fs=2.
            ), dtype=keras.config.floatx()
        )

        self.b = sos[:, :3]
        self.a = ops.stack(
            [
                ops.concatenate(
                    [
                        -subsec[None, -2:],
                        ops.eye(1, 2, 0)
                    ], axis=0
                ) for subsec in sos
            ], axis=0
        )

    # Notation from “Prefix Sums and Their Applications.” [Blelloch, Guy E.
    # 1990.]
    def _step_associative(self, s0: Tensor, ci: Tensor) -> Tensor:
        y0, x0 = s0
        ai, bi = ci

        yi = ops.matmul(ai, y0)  # ops.einsum("lij,ljk->lik", ai, y0)
        xi = ops.einsum("lsn,lbnd->lbsd", ai, x0) + bi

        return yi, xi

    def _filter_associative(
            self,
            bx: Tensor, a: Tensor,
            pad_mode: str, reverse: bool
    ) -> Tensor:
        # ICs
        match pad_mode:
            case "wrap":
                if reverse:
                    bx = ops.concatenate(
                        [
                            bx[:-1],
                            ops.einsum(
                                "sn, nbd -> bsd",
                                a,
                                bx[:2, :, 0, :]
                            )[None, ...] + bx[-1:]
                        ], axis=0
                    )
                else:
                    bx = ops.concatenate(
                        [
                            ops.einsum(
                                "sn, nbd -> bsd",
                                a,
                                bx[-1:-3:-1, :, 0, :]
                            )[None, ...] + bx[:1],
                            bx[1:]
                        ], axis=0
                    )
            case "edge":
                if reverse:
                    bx = ops.concatenate(
                        [
                            bx[:-1],
                            ops.einsum(
                                "sn, nbd -> bsd",
                                a,
                                ops.repeat(bx[-1:, :, 0, :], repeats=2, axis=0)
                            )[None, ...] + bx[-1:]
                        ], axis=0
                    )
                else:
                    bx = ops.concatenate(
                        [
                            ops.einsum(
                                "sn, nbd -> bsd",
                                a,
                                ops.repeat(bx[:1, :, 0, :], repeats=2, axis=0)
                            )[None, ...] + bx[:1],
                            bx[1:]
                        ], axis=0
                    )
            case "constant":
                pass

        return ops.swapaxes(
            ops.associative_scan(
                f=self._step_associative,
                elems=(ops.tile(a, [bx.shape[0], 1, 1]), bx),
                axis=0,
                reverse=reverse
            )[1], 0, 2
        )[0]

    def _step_iterative(
            self,
            y0: Tensor, bx: Tensor,
            a: Tensor) -> Tensor:
        aux = ops.matmul(a, y0) + bx
        return aux, aux[:, 0, :]

    def _filter_iterative(
            self,
            bx: Tensor, a: Tensor,
            pad_mode: str, reverse: bool
    ) -> Tensor:
        # ICs
        match pad_mode:
            case "wrap":
                if reverse:
                    carry = ops.swapaxes(bx[:2, :, 0, :], 0, 1)
                else:
                    carry = ops.swapaxes(bx[-1:-3:-1, :, 0, :], 0, 1)
            case "edge":
                if reverse:
                    carry = ops.swapaxes(
                        ops.repeat(bx[-1:, :, 0, :], repeats=2, axis=0),
                        0, 1
                    )
                else:
                    carry = ops.swapaxes(
                        ops.repeat(bx[:1, :, 0, :], repeats=2, axis=0),
                        0, 1
                    )
            case "constant":
                carry = ops.zeros([bx.shape[1], 2, bx.shape[-1]])

        return ops.swapaxes(
            ops.scan(
                f=partial(self._step_iterative, a=a),
                init=carry,
                xs=bx,
                reverse=reverse
            )[1], 0, 1
        )

    def _filt_filt(
            self,
            inputs: Tensor,
            pad_mode: str
    ) -> Tensor:
        if self.filter_mode == "iterative":
            filter_function = self._filter_iterative
        elif self.filter_mode == "associative":
            filter_function = self._filter_associative
        else:  # Invalid filter -> Do nothing
            return inputs

        l = inputs.shape[1]
        state_padding = ops.zeros_like(inputs)

        x = inputs

        for b, a in zip(self.b, self.a):
            for reverse in [False, True]:
                # causal padding input filter
                bx = ops.einsum(
                    "n,nbld->bld",
                    b,
                    ops.stack(
                        [
                            ops.pad(
                                x, [[0, 0], [0, i], [0, 0]], pad_mode
                            )[:, -l:, :]
                            if reverse else
                            ops.pad(
                                x, [[0, 0], [i, 0], [0, 0]], pad_mode
                            )[:, :l, :]
                            for i in range(3)
                        ], axis=0
                    )
                )

                # State-Padded Version
                bx = ops.swapaxes(
                    ops.stack(
                        [  # nbld
                            bx,
                            state_padding
                        ], axis=0
                    ), 0, 2
                )  # lbnd

                x = filter_function(bx, a, pad_mode, reverse)

        return x

    def call(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        # Double-pass to remove phase
        seasonality = self._filt_filt(inputs, DEFAULT_PAD_MODE)

        trend = inputs - seasonality

        return seasonality, trend


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrLayer(layers.Layer):
    """
    Keras implementation of the autocorrelation attention
    mechanism
    """

    def __init__(self, params: dict, **kwargs):
        super(CorrLayer, self).__init__(**kwargs)
        self.params = params

        self.k = params["k"]
        self.h = params["h"]

        self.d_keys: PositiveInt = params.get("d_keys", None)
        self.d_values: PositiveInt = params.get("d_values", None)

        self.dropout = layers.Dropout(params.get("dropout_rate", 0.))

        self.output_attention = params.get("output_attention", False)

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

    def _time_delay_agg_training(self, values: Tensor, corr: Tensor) -> Tensor:
        l = values.shape[-1]

        # Find top K (Batch-normalized)
        top_k = min(int(self.k * math.log(l)), l)
        mean_value = ops.mean(corr, axis=(1, 2))
        delay = ops.top_k(ops.mean(mean_value, axis=0), top_k)[1]

        # Nonlinear weights for aggregation
        tmp_corr = ops.softmax(
            mean_value[:, delay],
            axis=-1
        )[..., None, None, None]

        # Aggregation
        def roll_and_scale(idx: int, acc: Tensor) -> Tensor:
            return acc + tmp_corr[:, idx, ...] * values[
                ..., (delay[idx] + ops.arange(l)) % l
            ]
        delays_agg = ops.fori_loop(
            0, top_k, roll_and_scale, ops.zeros_like(values)
        )

        return delays_agg

    def _time_delay_agg_inference(self, values: Tensor, corr: Tensor) -> Tensor:
        l = values.shape[-1]

        # Find top K (Batch-wise)
        top_k = min(int(self.k * math.log(l)), l)
        mean_value = ops.mean(corr, axis=(1, 2))
        weights, delay = ops.top_k(mean_value, top_k)

        # Nonlinear weights for aggregation
        tmp_corr = ops.softmax(weights, axis=-1)[..., None, None, None]

        # Aggregation
        def roll_and_scale(idx: int, acc: Tensor) -> Tensor:
            return acc + tmp_corr[:, idx, ...] * ops.vectorized_map(
                lambda e: e[0][..., (e[1] + ops.arange(l)) % l],
                (values, delay[:, idx])
            )
        delays_agg = ops.fori_loop(
            0, top_k, roll_and_scale, ops.zeros_like(values)
        )
        return delays_agg

    def call(
        self,
        inputs: tuple[Tensor, Tensor, Tensor],
        training=None
    ) -> tuple[Tensor, Tensor]:
        queries, keys, values = inputs

        _, q_len, _ = queries.shape
        _, k_len, _ = keys.shape
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

        # Ensure dimension compatibility
        if q_len > k_len:
            zeros = ops.zeros_like(queries[:, :(q_len - k_len), ...])
            keys = ops.concatenate((keys, zeros), axis=1)
            values = ops.concatenate((values, zeros), axis=1)
        else:
            keys = keys[:, :q_len, :, :]
            values = values[:, :q_len, :, :]

        # Period-based dependencies
        q_fft = ops.rfft(ops.transpose(queries, (0, 2, 3, 1)))
        k_fft = ops.rfft(ops.transpose(keys, (0, 2, 3, 1)))
        corr = ops.irfft(
            (q_fft[0] * k_fft[0], -q_fft[1] * k_fft[1]),
            fft_length=q_len
        )

        if training:
            tda = self._time_delay_agg_training
        else:
            tda = self._time_delay_agg_inference
        out = ops.transpose(
            tda(
                ops.transpose(values, (0, 2, 3, 1)),
                corr
            ), axes=(0, 3, 1, 2)
        )
        out = self.out_proj(ops.reshape(out, (-1, q_len, h * self.d_values)))

        if self.output_attention:
            return (
                self.dropout(out),
                ops.transpose(
                    corr,
                    axes=(0, 3, 1, 2)
                )
            )
        else:
            return self.dropout(out), None


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrEncoderLayer(layers.Layer):
    """
    Keras implementation of a single Autoformer encoder layer
    """

    def __init__(self, params: dict, **kwargs):
        super(CorrEncoderLayer, self).__init__(**kwargs)
        self.params = params

        self.d_ff: int = params.get("d_ff", None)
        self.activation = params.get("activation", "relu")
        self.tau: int = params["tau"]

        decomposition = params.get("decomposition", "iir")
        if decomposition == "ma":
            self.decomposition_layer = MADecompositionLayer(self.tau)
        elif decomposition == "iir":
            self.decomposition_layer = IIRDecompositionLayer(self.tau)

        self.attn_layer = CorrLayer(params)

        self.dropout = layers.Dropout(params.get("dropout_rate", 0.))
        self.ff_layer: keras.Sequential = None

    def build(self, input_shape: tuple[int, ...]):
        self.d_ff = self.d_ff or 4 * input_shape[-1]

        self.ff_layer = keras.Sequential(
            [
                layers.Dense(
                    units=self.d_ff,
                    use_bias=False,
                    activation=self.activation
                ),
                self.dropout,
                layers.Dense(
                    units=input_shape[-1],
                    use_bias=False,
                    activation=None
                ),
                self.dropout
            ]
        )

    def call(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        x, attn = self.attn_layer([inputs, inputs, inputs])
        x = inputs + self.dropout(x)
        x, _ = self.decomposition_layer(x)

        y = self.ff_layer(x)
        y, _ = self.decomposition_layer(x + y)

        return y, attn

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrEncoder(layers.Layer):
    """
    Keras implementation of the Autoformer encoder block
    """

    def __init__(self, params: dict, **kwargs):
        super(CorrEncoder, self).__init__(**kwargs)
        self.params = params

        n_layers = params["N"]

        self.embed = DataEmbedding(
            params["d_model"],
            params["dropout_rate"],
            params["embed_type"],
            params.get("freq")
        )

        if isinstance(params["tau"], Sequence):
            if len(params["tau"]) < n_layers:
                self.enc_layers = [
                    CorrEncoderLayer({**params, "tau": tau})
                    for tau, _ in zip_longest(
                        params["tau"],
                        range(n_layers),
                        fillvalue=params["tau"][-1]
                    )
                ]
            else:
                self.enc_layers = [
                    CorrEncoderLayer({**params, "tau": tau})
                    for tau, _ in zip(
                        params["tau"][-n_layers:],
                        range(n_layers)
                    )
                ]
        else:
            self.enc_layers = [
                CorrEncoderLayer(params)
                for _ in range(n_layers)
            ]

        if params["normalize"]:
            self.norm_layer = keras.Sequential([
                layers.LayerNormalization(axis=-1),
                layers.LayerNormalization(axis=-2, scale=False)
            ])
        else:
            self.norm_layer = layers.Identity()

    def build(self, input_shape):
        embedded_shape = [*input_shape[0][:-1], self.params["d_model"]]
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

        return self.norm_layer(x), tuple(attns)

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrDecoderLayer(layers.Layer):
    """
    Keras implementation of a single Autoformer decoder layer
    """

    def __init__(self, params: dict, **kwargs):
        super(CorrDecoderLayer, self).__init__(**kwargs)
        self.params = params

        self.d_ff: int = params.get("d_ff", None)
        self.activation = params.get("activation", "softplus")
        self.tau = params["tau"]

        decomposition = params.get("decomposition", "iir")
        if decomposition == "ma":
            self.decomposition_layer = MADecompositionLayer(self.tau)
        elif decomposition == "iir":
            self.decomposition_layer = IIRDecompositionLayer(self.tau)

        self.dropout = layers.Dropout(params.get("dropout_rate", 0.))
        self.self_attn = CorrLayer({**params, "output_attention": False})
        self.cross_attn = CorrLayer({**params, "output_attention": False})

        self.ff_layer: keras.Sequential = None

        self.out_proj = layers.Conv1D(
            filters=params["d_enc"], kernel_size=3,
            strides=1, padding="same",
            use_bias=False,
            data_format="channels_last"
        )

    def build(self, input_shape: tuple[tuple[int, ...], ...]):
        self.d_ff = self.d_ff or 4 * input_shape[0][-1]

        self.ff_layer = keras.Sequential(
            [
                layers.Dense(
                    units=self.d_ff, use_bias=False,
                    activation=self.activation
                ),
                self.dropout,
                layers.Dense(
                    units=input_shape[0][-1], use_bias=False,
                    activation=None
                ),
                self.dropout
            ]
        )

    def call(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, cross = inputs

        x += self.dropout(self.self_attn([x, x, x])[0])
        x, xt_1 = self.decomposition_layer(x)

        x += self.dropout(self.cross_attn([x, cross, cross])[0])
        x, xt_2 = self.decomposition_layer(x)

        y = self.ff_layer(x)
        y, xt_3 = self.decomposition_layer(x + y)

        residual_trend = self.out_proj(xt_1 + xt_2 + xt_3)

        return y, residual_trend

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrDecoder(layers.Layer):
    """
    Keras implementation of the Autoformer decoder block
    """

    def __init__(self, params: dict, **kwargs):
        super(CorrDecoder, self).__init__(**kwargs)
        self.params = params

        d = params["d_enc"]
        n_layers = params["M"]

        self.o = params["O"]
        self.output_components = params["output_components"]

        self.embed = DataEmbedding(
            params["d_model"],
            params["dropout_rate"],
            params["embed_type"],
            params.get("freq")
        )

        if isinstance(params["tau"], Sequence):
            if len(params["tau"]) < n_layers:
                self.dec_layers = [
                    CorrDecoderLayer({**params, "tau": tau})
                    for tau, _ in zip_longest(
                        params["tau"],
                        range(n_layers),
                        fillvalue=params["tau"][-1]
                    )
                ]
            else:
                self.dec_layers = [
                    CorrDecoderLayer({**params, "tau": tau})
                    for tau, _ in zip(
                        params["tau"][-n_layers:],
                        range(n_layers)
                    )
                ]
        else:
            self.dec_layers = [
                CorrDecoderLayer(params)
                for _ in range(n_layers)
            ]

        if params["normalize"]:
            self.norm_layer = keras.Sequential([
                layers.LayerNormalization(axis=-1),
                layers.LayerNormalization(axis=-2, scale=False)
            ])
        else:
            self.norm_layer = layers.Identity()

        self.out_proj = layers.Dense(
            d, activation="linear",
            use_bias=True
        )

    def build(self, input_shape: tuple[tuple[int], ...]):
        embedded_shape = input_shape[-1]
        for decoder_layer in self.dec_layers:
            decoder_layer.build((embedded_shape, embedded_shape))

    def call(
        self,
        inputs: tuple[Tensor | tuple[Tensor, ...], ...]
    ) -> tuple[Tensor, Tensor]:
        (xs, xt), xm, cross = inputs
        xs = self.embed([xs, xm])

        for dec_layer in self.dec_layers:
            xs, residual_trend = dec_layer([xs, cross])
            xt += residual_trend

        xs = self.out_proj(self.norm_layer(xs))

        xs = xs[:, -self.o:, :]
        xt = xt[:, -self.o:, :]

        if self.output_components:
            return xs + xt, xs, xt
        else:
            return xs + xt

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


@keras.saving.register_keras_serializable(package="Autoformer")
class CorrDecoderInit(layers.Layer):
    """
    Adapter layer for the seasonal/trend mechanism of the
    Autoformer.
    """

    def __init__(self, params: dict, **kwargs):
        super(CorrDecoderInit, self).__init__(**kwargs)
        self.params = params

        # Whether to adapt a manual input (known future input) or
        # generate a placeholder (paper strategy)
        self.manual_dec_input = params["manual_dec_input"]

        # moving average window
        if isinstance(params["tau"], Sequence):
            self.tau = params["tau"][0]
        else:
            self.tau = params["tau"]

        decomposition = params.get("decomposition", "iir")
        if decomposition == "ma":
            self.decomposition_layer = MADecompositionLayer(self.tau)
        elif decomposition == "iir":
            self.decomposition_layer = IIRDecompositionLayer(self.tau)

        self.o = params["O"]  # prediction horizon

    def call(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        if self.manual_dec_input:
            x_enc, x_enc_marks, x_dec, x_dec_marks = inputs
            _, l_enc, d_enc = x_enc.shape
            _, l_dec, d_dec = x_dec.shape

            x_dec = ops.concatenate(
                (
                    x_enc[:, l_enc // 2:, :],
                    ops.concatenate(
                        (
                            x_dec,
                            ops.mean(x_enc[..., d_dec:], 1, True) *
                            ops.ones((l_dec, d_enc - d_dec))[None, ...]
                        ), axis=-1
                    )
                ), axis=1
            )
            x_dec_s, x_dec_t = self.decomposition_layer(x_dec)
        else:
            x_enc, x_enc_marks, x_dec_marks = inputs
            l_enc = x_enc.shape[1]
            l_dec = x_dec_marks.shape[1]

            half_season, half_trend = self.decomposition_layer(
                x_enc[:, l_enc // 2:, :]
            )

            x_dec_s = ops.pad(
                half_season, ((0, 0), (0, self.o), (0, 0)),
                mode="constant", constant_values=0
            )
            x_dec_t = ops.concatenate(
                (
                    half_trend,
                    ops.mean(x_enc, 1, True) *
                    ops.ones((l_dec,))[None, :, None]
                ), axis=1
            )

        x_dec_marks = ops.concatenate(
            (
                x_enc_marks[:, l_enc // 2:, :],
                x_dec_marks
            ), axis=1
        )

        return x_dec_s, x_dec_t, x_dec_marks

    def build(self, input_shape: tuple[tuple[int, ...], ...]):
        b, _, d = input_shape[0]
        i, o = self.params["I"], self.params["O"]
        self.decomposition_layer.build((b, i + (o - o // 2), d))

    def get_config(self):
        config = super().get_config()

        config.update({
            "params": self.params
        })

        return config


def restore_custom_objects():
    keras.saving.get_custom_objects().update(
        {
            "Autoformer>MADecompositionLayer": MADecompositionLayer,
            "Autoformer>IIRDecompositionLayer": IIRDecompositionLayer,
            "Autoformer>CorrLayer": CorrLayer,
            "Autoformer>CorrEncoderLayer": CorrEncoderLayer,
            "Autoformer>CorrEncoder": CorrEncoder,
            "Autoformer>CorrDecoderLayer": CorrDecoderLayer,
            "Autoformer>CorrDecoder": CorrDecoder,
            "Autoformer>CorrDecoderInit": CorrDecoderInit
        }
    )
