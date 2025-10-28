"""
Python module implementing the layers of a Canonical
Transformer in Keras.
"""

from collections.abc import Sequence

import keras
from keras import Input
from keras import constraints
from keras import KerasTensor as Tensor
from keras import layers

from . import embed
from . import autoformer
from . import informer
from . import recurrent

from pydantic import Field
from typing import Annotated, Optional

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]


@keras.saving.register_keras_serializable(package="GenericTransformer")
class GenericTransformer(layers.Layer):
    """
    Keras implementation of a Generic Transformer architecture
    """

    def __init__(
        self,
        encoder: layers.Layer,
        decoder: layers.Layer,
        output_attention: bool = False,
        **kwargs
    ):
        super(GenericTransformer, self).__init__(**kwargs)

        # wether to output the model attention
        self.output_attention = output_attention

        self.encoder = encoder
        self.decoder = decoder

    def call(
        self,
        inputs: tuple[Tensor | tuple[Tensor], ...]
    ) -> Tensor | tuple[Tensor | tuple[Tensor], ...]:
        # Enc Input -> Dec Input
        x_enc, x_enc_marks, x_dec, x_dec_marks = inputs

        # Enc call
        y_enc, attns = self.encoder([x_enc, x_enc_marks])
        # Dec call
        y = self.decoder([x_dec, x_dec_marks, y_enc])

        if isinstance(y, Sequence):
            y = list(y)
        else:
            y = [y]

        if self.output_attention:
            y.append(keras.ops.stack(attns, axis=2))

        if len(y) == 1:
            y = y[0]

        return y

    def get_config(self):
        config = super().get_config()

        config.update({
            "output_attention": self.output_attention,
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder)
        })

        return config

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop("encoder")
        encoder = keras.saving.deserialize_keras_object(encoder_config)

        decoder_config = config.pop("decoder")
        decoder = keras.saving.deserialize_keras_object(decoder_config)

        return cls(encoder, decoder, **config)


def get_mark_inputs(params: dict) -> tuple[Tensor, Tensor]:
    if params["embed_type"] is None:
        x_enc_marks = 0
        x_dec_marks = 0
    else:
        if params["embed_type"] == "temporal":
            # Hardcoded '4 (5)' for [Month, DoM, DoW, Hour, (Minute)]
            if params["freq"] == "h":
                x_enc_marks_shape = (params["I"], 4)
                x_dec_marks_shape = (params["O"], 4)
            elif params["freq"] == "m":
                x_enc_marks_shape = (params["I"], 5)
                x_dec_marks_shape = (params["O"], 5)
            else:
                x_enc_marks_shape = (params["I"], 0)
                x_dec_marks_shape = (params["O"], 0)
        else:
            x_enc_marks_shape = (params["I"],)
            x_dec_marks_shape = (params["O"],)

        x_enc_marks = Input(
            shape=x_enc_marks_shape,
            name="xm_enc_input",
            dtype="uint32"
        )
        x_dec_marks = Input(
            shape=x_dec_marks_shape,
            name="xm_dec_input",
            dtype="uint32"
        )

    return x_enc_marks, x_dec_marks


def create_autoformer_model(
    *, I: int, O: int,
    d_model: int,
    d_ff: int,
    k: int, h: int,
    tau: int | Sequence[int],
    d_enc: int, d_dec: Optional[int] = None,
    decomposition: Optional[str] = "iir",
    d_out: Optional[int] = None,
    M: Optional[int] = None, N: Optional[int] = None,
    n_blocks: int = 1,
    dropout_rate: float = 0.,
    output_components: bool = False,
    output_attention: bool = False,
    embed_type: str = "temporal",
    freq: str = "h",
    manual_dec_input: bool = False,
    normalize: bool = False,
    **_
) -> keras.Model:
    """Create an Autoformer model from a parameter list.

    Helper function that takes an argument list an instantiates the necesary
    layers and inputs for an Autoformer architecture using the Functional
    interface from the Keras library.

    Args:
      I:
        Encoder Input sequence length.
      O:
        Decoder Input and Output sequence length.
      d:
        Number of Encoder/Decoder Input channels. When using a manual decoder
        input, remember that these two tensors must have the same dimension and
        channel order, since part of the encoder input is prepended to the
        decoder input.
      d_model:
        The model's embedding depth.
      d_ff:
        Dimension of units/neurons per Feed-Forward layer.
      k:
        Sampling factor for the ProbAttention mechanism.
      h:
        Number of attention heads.
      tau:
        Integer or Sequence of integers determining the average-window
        lengths.
      decomposition:
        Method by which the Seasonal-Trend decomposition is performed.
      d_out:
        Number of Output channels. Defaults to the Input dimensionality.
      N:
        Number of Encoder Blocks.
      M:
        Number of Decoder Blocks.
      n_blocks:
        Alternative way to define the same number for Encoder/Decoder block
        count.
      dropout_rate:
        Self-explanatory, applied between attention and feed-forward layers.
      output_components:
        Boolean flag indicating whether to output the season/trend decomposition
        from which the output is derived.
      output_attention:
        Boolean flag indicating whether to output the attention tensor from the
        Encoder stack.
      embed_type:
        String indicating which kind ("temporal", "fixed", "positional",
        or None) of embedding to use.
      freq:
        In the case of temporal encoding, whether the minute stamps are included
        in the marks' Tensor.
      manual_dec_input:
        Boolean flag indicating whether the decoder input is manually provided
        (when future information is known) or initialized with the placeholder
        formulated by the original paper.
      normalize:
        Boolean flag indicating whether to normalize the sequences at every
        encoder/decoder block.

    Returns:
      A Keras Model instance representing the Informer model instance.
    """

    restore_custom_objects()

    params = {
        "I": I,
        "O": O,
        "d_enc": d_enc,
        "d_dec": d_dec or d_enc,
        "d_model": d_model,
        "d_ff": d_ff,
        "M": M or n_blocks,
        "N": N or n_blocks,
        "k": k,
        "h": h,
        "tau": tau,
        "decomposition": decomposition,
        "dropout_rate": dropout_rate,
        "d_out": d_out or d_enc,
        "output_components": output_components,
        "output_attention": output_attention,
        "embed_type": embed_type,
        "freq": freq,
        "manual_dec_input": manual_dec_input,
        "normalize": normalize
    }

    x_enc_shape = (params["I"], params["d_enc"])
    x_dec_shape = (params["O"], params["d_dec"])

    x_enc = Input(
        shape=x_enc_shape,
        name="x_enc_input"
    )
    x_enc_marks, x_dec_marks = get_mark_inputs(params)

    if params["manual_dec_input"]:
        x_dec = Input(
            shape=x_dec_shape,
            name="x_dec_input"
        )
        x_dec_s, x_dec_t, x_dec_marks_ext = \
            autoformer.CorrDecoderInit(params)(
                [x_enc, x_enc_marks, x_dec, x_dec_marks]
            )
    else:
        x_dec_s, x_dec_t, x_dec_marks_ext = \
            autoformer.CorrDecoderInit(params)(
                [x_enc, x_enc_marks, x_dec_marks]
            )

    if isinstance(params["tau"], Sequence):
        params["tau"] = sorted(
            params["tau"],
            reverse=True
        )

    encoder = autoformer.CorrEncoder(params)
    decoder = autoformer.CorrDecoder(params)
    y_af = GenericTransformer(
        encoder,
        decoder,
        params["output_attention"]
    )(
        [
            x_enc, x_enc_marks,
            [x_dec_s, x_dec_t], x_dec_marks_ext
        ]
    )

    if params["d_out"] != x_enc_shape[-1]:
        adapter = layers.Dense(
            params["d_out"], use_bias=False,
            kernel_constraint=constraints.UnitNorm()
        )
        if params["output_components"] or params["output_attention"]:
            y = [adapter(y_af[0])]
            if params["output_components"]:
                y.extend([adapter(y_af[1]), adapter(y_af[2])])
            if params["output_attention"]:
                y.append(y_af[-1])
        else:
            y = adapter(y_af)
    else:
        y = y_af

    if params["manual_dec_input"]:
        if params["embed_type"] is None:
            return keras.Model(
                inputs=[x_enc, x_dec],
                outputs=y
            )
        else:
            return keras.Model(
                inputs=[
                    x_enc, x_enc_marks,
                    x_dec, x_dec_marks
                ],
                outputs=y
            )
    else:
        if params["embed_type"] is None:
            return keras.Model(
                inputs=x_enc,
                outputs=y
            )
        else:
            return keras.Model(
                inputs=[
                    x_enc, x_enc_marks,
                    x_dec_marks
                ],
                outputs=y
            )


def create_informer_model(
    *, I: int, O: int,
    d_enc: int, d_dec: int, d_out: int,
    d_model: int = 512, d_ff: int = 512,
    k: int = 5, h: int = 8,
    N: Optional[int] = None, M: Optional[int] = None,
    n_blocks: int = 1,
    dropout_rate: float = 0.,
    output_attention: bool = False,
    embed_type: str = "positional",
    freq: str = "h",
    normalize: bool = False,
    **_
) -> keras.Model:
    """Create an Informer model from a parameter list.

    Helper function that takes an argument list an instantiates the necesary
    layers and inputs for an Informer architecture using the Functional
    interface from the Keras library.

    Args:
      I:
        Encoder Input sequence length.
      O:
        Decoder Input and Output sequence length.
      d_enc:
        Number of Encoder Input channels.
      d_dec:
        Number of Decoder Input channels.
      d_out:
        Number of Output channels.
      d_model:
        The model's embedding depth.
      d_ff:
        Dimension of units/neurons per Feed-Forward layer.
      k:
        Sampling factor for the ProbAttention mechanism.
      h:
        Number of attention heads.
      N:
        Number of Encoder Blocks.
      M:
        Number of Decoder Blocks.
      n_blocks:
        Alternative way to define the same number for Encoder/Decoder block
        count.
      dropout_rate:
        Self-explanatory, applied between attention and feed-forward layers.
      output_attention:
        Boolean flag indicating whether to output the attention tensor from the
        Encoder stack.
      embed_type:
        String indicating which kind ("temporal", "fixed", "positional",
        or None) of embedding to use. "None" not recommended for
        this architecture.
      freq:
        In the case of temporal encoding, whether the minute stamps are included
        in the marks' Tensor.
      normalize:
        Boolean flag indicating whether to normalize the sequences at every
        encoder/decoder block.
    Returns:
      A Keras Model instance representing the Informer model instance.
    """

    restore_custom_objects()

    params = {
        "I": I,
        "O": O,
        "d_enc": d_enc,
        "d_dec": d_dec,
        "d_out": d_out,
        "d_model": d_model,
        "d_ff": d_ff,
        "k": k,
        "h": h,
        "N": N or n_blocks,
        "M": M or n_blocks,
        "dropout_rate": dropout_rate,
        "output_attention": output_attention,
        "embed_type": embed_type,
        "freq": freq,
        "normalize": normalize
    }

    x_enc = Input(
        shape=(params["I"], params["d_enc"]),
        name="x_enc_input"
    )
    x_dec = Input(
        shape=(params["O"], params["d_dec"]),
        name="x_dec_input"
    )
    x_enc_marks, x_dec_marks = get_mark_inputs(params)

    y = GenericTransformer(
        informer.ProbEncoder(params),
        informer.ProbDecoder(params),
        params["output_attention"]
    )(
        [
            x_enc, x_enc_marks,
            x_dec, x_dec_marks
        ]
    )

    if params["embed_type"] is None:
        return keras.Model(
            inputs=[x_enc, x_dec],
            outputs=y
        )
    else:
        return keras.Model(
            inputs=[
                x_enc, x_enc_marks,
                x_dec, x_dec_marks
            ],
            outputs=y
        )


def create_recurrent_model(
    *, I: int, O: int,
    d_enc: int, d_dec: int, d_out: int,
    d_model: int = 512, d_ff: int = 512,
    h: int = 16,
    N: Optional[int] = None, M: Optional[int] = None,
    n_blocks: int = 1,
    rnn_type: str = "lstm",
    dropout_rate: float = 0.,
    output_attention: bool = False,
    embed_type: str = None,
    freq: str = "h",
    normalize: bool = False,
    **_
) -> keras.Model:
    """Create a Sequence2Sequence Encoder-Decoder model using recurrent kernels

    Helper function that takes an argument list an instantiates the necesary
    layers and inputs for an Encoder-Decoder recurrent architecture using the
    Functional interface from the Keras library.

    Args:
      I:
        Encoder Input sequence length.
      O:
        Decoder Input and Output sequence length.
      d_enc:
        Number of Encoder Input channels.
      d_dec:
        Number of Decoder Input channels.
      d_out:
        Number of Output channels.
      d_model:
        Model embedding depth.
      d_ff:
        Dimension of units/neurons per Feed-Forward layer.
      h:
        Recurrent units (heads) per block.
      N:
        Number of Encoder Blocks.
      M:
        Number of Decoder Blocks.
      n_blocks:
        Alternative way to define the same number for Encoder/Decoder block
        count.
      dropout_rate:
        Self-explanatory, applied between attention and feed-forward layers.
      output_attention:
        Boolean flag indicating whether to output the attention tensor from the
        Encoder stack.
      embed_type:
        String indicating which kind ("temporal", "fixed", "positional",
        or None) of embedding to use.
      freq:
        In the case of temporal encoding, whether the minute stamps are included
        in the marks' Tensor.
      normalize:
        Boolean flag indicating whether to normalize the sequences at every
        encoder/decoder block.
    Returns:
      A Keras Model instance representing the Recurrent model instance.
    """

    restore_custom_objects()

    params = {
        "I": I,
        "O": O,
        "d_enc": d_enc,
        "d_dec": d_dec,
        "d_out": d_out,
        "d_model": d_model,
        "d_ff": d_ff,
        "h": h,
        "N": N or n_blocks,
        "M": M or n_blocks,
        "rnn_type": rnn_type,
        "dropout_rate": dropout_rate,
        "output_attention": output_attention,
        "embed_type": embed_type,
        "freq": freq,
        "normalize": normalize
    }

    x_enc = Input(
        shape=(params["I"], params["d_enc"]),
        name="x_enc_input"
    )
    x_dec = Input(
        shape=(params["O"], params["d_dec"]),
        name="x_dec_input"
    )

    x_enc_marks, x_dec_marks = get_mark_inputs(params)

    y = GenericTransformer(
        recurrent.RecEncoder(params),
        recurrent.RecDecoder(params),
        params["output_attention"]
    )(
        [
            x_enc, x_enc_marks,
            x_dec, x_dec_marks
        ]
    )

    if params["embed_type"] is None:
        return keras.Model(
            inputs=[x_enc, x_dec],
            outputs=y
        )
    else:
        return keras.Model(
            inputs=[
                x_enc, x_enc_marks,
                x_dec, x_dec_marks
            ],
            outputs=y
        )


def restore_custom_objects():
    keras.saving.get_custom_objects().update({
        "GenericTransformer>GenericTransformer": GenericTransformer
    })

    embed.restore_custom_objects()
    autoformer.restore_custom_objects()
    recurrent.restore_custom_objects()
