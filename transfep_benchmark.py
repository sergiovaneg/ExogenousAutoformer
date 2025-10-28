"""
Script to train and print the baseline results
"""

import os
import sys
import time

import json

from functools import partial
from keras.ops import scan

import numpy as np
from scipy.stats import ttest_ind

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
import transfep_utils

import keras
from keras import KerasTensor as Tensor
from keras import ops

from keras_transformer import generic_transformer as transformer

keras.utils.set_random_seed(1)
args = transfep_utils.get_cli_args()
transformer.restore_custom_objects()

sns.set_theme(context="paper",
              style="whitegrid",
              font="Roboto",
              font_scale=2)

REPS = 5
EPOCHS = 1000
BATCH_SIZE: int = args.batch_size

I: int = args.context_window
O: int = args.prediction_horizon
Y: str = args.target

ROOT = args.root
DATABASE = "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/transfep"
STUDY_FILEPATH: str = args.study

LIN_THR: float = 1E0 if Y == "energy" else 1E-2

ENABLE_ITERATIVE = False

with open(
    "./params/variables.json",
    "r",
    encoding=sys.getdefaultencoding()
) as f:
    VAR_JSON = json.load(f)


def naive_predict(x: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.tile(
        x[0][:, -24:, -1],
        [1, 3]
    ).reshape([-1, O, 1])


def iter_step(
    y0: Tensor,
    xs: tuple[Tensor, ...],
    m: keras.Model
) -> tuple[Tensor, Tensor]:
    xs[2] = ops.concatenate(
        [
            xs[2][:, :-1],
            ops.concatenate(
                [
                    y0[24:, :],
                    ops.tile(
                        ops.mean(ops.concatenate(
                            [
                                xs[0][:, -1:],
                                y0
                            ], axis=0
                        )),
                        [24, 1]
                    )
                ], axis=0
            )
        ], axis=-1
    )
    y = m([e[None, ...] for e in xs])[0, ...]

    return y, y


def iter_predict_lax(x: tuple[Tensor, ...], m: keras.Model) -> Tensor:
    return scan(
        partial(iter_step, m=m),
        ops.tile(  # Naive Init
            x[0][0, -24:, -1:],
            [3, 1]
        ),
        x
    )[1]


def iter_predict(x: tuple[Tensor, ...], m: keras.Model) -> Tensor:
    carry = ops.tile(  # Naive Init
        x[0][0, -24:, -1:],
        [3, 1]
    )
    ys = []

    for row in zip(*x):
        carry, y = iter_step(carry, list(row), m)
        ys.append(y)

    return ops.stack(ys)


def calc_rms(arr):
    return np.sqrt(np.mean(np.square(arr)))


def make_histogram(
    ys: list[np.ndarray],
    yhats: list[np.ndarray]
) -> Figure:
    errors, horizons = [], []
    for offset, y, yhat in zip([0, 24, 48], ys, yhats):
        errors.append(yhat - y)
        horizons.append(
            np.tile(
                np.arange(
                    offset +
                    1,
                    offset +
                    25),
                len(y) //
                24))

    error = np.concatenate(errors)
    horizon = np.concatenate(horizons)

    lin_thr_log = np.log10(LIN_THR)

    err_bins = np.concatenate(
        [
            -np.logspace(lin_thr_log, lin_thr_log + 3, 4)[::-1],
            np.logspace(lin_thr_log, lin_thr_log + 3, 4)
        ]
    )

    fig = plt.figure(figsize=(12, 6))
    plt.hist2d(
        horizon,
        error,
        [72, err_bins],
        cmap="viridis",
        norm="log"
    )
    plt.xlabel(r"Prediction Horizon $[h]$")
    plt.ylabel(r"Error $[kW]$")
    plt.yscale("symlog", linthresh=LIN_THR, linscale=0.5)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Frequency")
    fig.tight_layout(pad=2)

    return fig


def process_results(
    r: dict[
        str,
        tuple[
            list[np.ndarray],
            list[float]
        ]
    ]
):
    # Enforce order while checking if archs are available
    model_keys = [
        arch for arch in [
            "Seasonal-Naive",
            "RNN",
            "Informer",
            "Autoformer",
            "AF++",
            "AF++ Iterative"
        ] if arch in r.keys()
    ]
    ref = r["Reference"]

    aux_err = np.sum(
        np.stack(
            [
                np.concatenate(
                    [
                        np.abs(
                            ref[0][day_idx] - r[arch][0][day_idx]
                        ).reshape([-1, 24])
                        for day_idx in range(3)
                    ], axis=-1
                )
                for arch in model_keys
            ], axis=0
        ), axis=(0, -1)
    )

    best_idx, worst_idx = 24 * np.argmin(aux_err), 24 * np.argmax(aux_err)

    figs = [plt.figure(figsize=(12, 6)) for _ in range(2)]
    x_range = np.arange(1, 73, 1, dtype=int)
    for arch in ["Reference"] + model_keys:
        for fig, target_idx in zip(figs, [best_idx, worst_idx]):
            aux = np.concatenate(
                [yhat[target_idx:target_idx + 24]
                 for yhat in r[arch][0]]
            )
            ax = fig.gca()
            if arch == "Reference":
                ax.plot(x_range, aux, "--", label=arch)
            elif arch.startswith("AF++"):
                ax.plot(x_range, aux, linewidth=2., label=arch)
            else:
                ax.plot(x_range, aux, label=arch)

    for fig, fig_label in zip(figs, ["best", "worst"]):
        ax = fig.gca()
        ax.legend()
        ax.set_xlabel(r"Prediction Horizon $[h]$")
        ax.set_ylabel(r"Power Consumption $[kW]$")
        fig.tight_layout(pad=2)
        fig.savefig(f"./figures/{Y}/forecasts_I{I}_{fig_label}.svg")

    latex_percent = ""
    latex_significance = ""
    latex_time = ""

    afpp_err_ld = r["AF++"][0][-1] - ref[0][-1]
    afpp_time = np.median(r["AF++"][1])

    for arch in model_keys:
        arch_time = np.median(r[arch][1])

        print(arch)
        print(
            " & ".join(
                [
                    f"${calc_rms(
                        np.stack(
                            [
                                y - yhat
                                for y, yhat in zip(ref[0], r[arch][0])
                            ]
                        )
                    ):.3}$"
                ]
                +
                [
                    f"${calc_rms(y - yhat):.3}$"
                    for y, yhat in zip(ref[0], r[arch][0])
                ] + [
                    f"${arch_time:.3}$"
                ]
            )
        )

        fig = make_histogram(ref[0], r[arch][0])
        fig.savefig(f"./figures/{Y}/error_hist_{arch}_I{I}.svg")

        if not arch.startswith("AF++"):
            arch_err_ld = r[arch][0][-1] - ref[0][-1]
            latex_percent += f"""& ${
                100 * (
                    calc_rms(arch_err_ld) - calc_rms(afpp_err_ld)
                ) / calc_rms(arch_err_ld):.3}\\%$ """

            ttest_results = ttest_ind(
                arch_err_ld,
                afpp_err_ld,
                equal_var=False
            )
            latex_significance += f"& ${ttest_results.pvalue:.3}$ "

            latex_time += f"""& ${
                100 * (arch_time - afpp_time) / arch_time:.3}\\%$ """

    print(latex_percent)
    print(latex_significance)
    print(latex_time)

    if ENABLE_ITERATIVE:
        afpp_iter_err_ld = r["AF++ Iterative"][0][-1] - ref[0][-1]
        afpp_err_global = np.concat(
            [
                yhat - y
                for y, yhat in zip(ref[0], r["AF++"][0])
            ]
        )
        afpp_iter_err_global = np.concat(
            [
                yhat - y
                for y, yhat in zip(ref[0], r["AF++ Iterative"][0])
            ]
        )

        print(f"""Fine-tune improvement: ${
            100 * (
              calc_rms(afpp_err_global) - calc_rms(afpp_iter_err_global)
              ) / calc_rms(afpp_err_global)
        }\\%$ & ${
            100 * (
                calc_rms(afpp_err_ld) - calc_rms(afpp_iter_err_ld)
            ) / calc_rms(afpp_err_ld)
        }\\%$""")

        print(f"""Fine-tune significance: ${
            ttest_ind(
                afpp_err_global,
                afpp_iter_err_global,
                equal_var=False
            ).pvalue
        }$ & ${
            ttest_ind(
                afpp_err_ld,
                afpp_iter_err_ld,
                equal_var=False
            ).pvalue
        }$""")


if __name__ == "__main__":
    relevant_variables = VAR_JSON[Y]
    loss, study_name, save_path = transfep_utils.get_loss_and_paths(args)

    best_trial = optuna.load_study(
        study_name=study_name,
        storage=DATABASE
    ).best_trial

    model_params: dict = best_trial.params
    model_params.update({
        "I": I,
        "O": O,
        "d": len(relevant_variables) + 1,
        "d_enc": len(relevant_variables) + 1,
        "d_dec": len(relevant_variables) + 1,
        "d_out": 1,
        "rnn_type": "gru",
        "manual_dec_input": True,
        "embed_type": "temporal",
        "freq": "h",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_batch_size": max(
            1,
            round(np.log(BATCH_SIZE))
        ),
        "learning_rate": 1e-3,
    })
    print(f"AF++ Training time: {
        (
            best_trial.datetime_complete -
            best_trial.datetime_start
        ).total_seconds():.3
    }", flush=True)

    raw_data = transfep_utils.get_curated_datasets(
        relevant_variables + [Y],
        # Using logarithmic loss instead of modifying the signal
        log_solar=False,
        root=ROOT
    )

    dataset = transfep_utils.dataset_formatter(
        raw_data=raw_data,
        model_params=model_params,
        target=Y,
        stride=1,
        updatable=False,
        iterative=False,
        update_stride=0,
        normalize=Y == "full_solar",
        context_days=30 if Y == "full_solar" else 0,
        sigma=0.001,
        beta=0.03
    )

    dataset["test"] = transfep_utils.dataset_formatter(
        raw_data=raw_data,
        model_params=model_params,
        target=Y,
        stride=24,
        updatable=False,
        iterative=False,
        update_stride=0,
        normalize=Y == "full_solar",
        context_days=30 if Y == "full_solar" else 0,
        sigma=0.001,
        beta=0.03
    )["test"]

    results = {
        "Reference": [
            [
                dataset["test"].y[:, s:s + 24, 0].flatten()
                for s in [0, 24, 48]
            ],
            [0.]
        ],
        "Seasonal-Naive": transfep_utils.benchmark_model(
            naive_predict,
            dataset["test"],
            row_wise=True,
            repetitions=REPS
        )
    }

    # Autoformer++ (single-pass)
    keras_model = keras.saving.load_model(
        os.path.join(
            save_path,
            "best.keras"
        ),
        compile=False
    )
    keras_model.summary()
    results["AF++"] = transfep_utils.benchmark_model(
        keras_model,
        dataset["test"],
        row_wise=True,
        repetitions=REPS
    )

    print("IT'S NOT THE AUTOFORMER.", flush=True)

    # Autoformer++ (iterative)
    if ENABLE_ITERATIVE:
        assert os.path.exists(
            os.path.join(
                save_path,
                "best_iterative.keras"
            )
        ), "Run 'transfep_finetune' first to get the iterative model"
        keras_model = keras.saving.load_model(
            os.path.join(
                save_path,
                "best_iterative.keras"
            ),
            compile=False
        )
        keras_model.summary()
        results["AF++ Iterative"] = transfep_utils.benchmark_model(
            partial(iter_predict, m=keras_model),
            dataset["test"],
            row_wise=False,  # Already iterative inside
            repetitions=True
        )

    # Remove full_solar placeholder in decoder input
    model_params["d_dec"] -= 1
    for partition, inputs in dataset.items():
        dataset[partition].x[2] = inputs.x[2][..., :-1]

    # Informer
    if not os.path.exists(
            os.path.join(save_path, "informer.keras")
    ):
        print("Started Informer training...", flush=True)
        keras_model = transformer.create_informer_model(**model_params)

        keras_model.compile(
            keras.optimizers.Adam(model_params["learning_rate"]),
            loss=loss
        )

        start_time = time.time()
        keras_model.fit(
            x=dataset["train"],
            epochs=EPOCHS,
            shuffle=False,
            validation_data=dataset["validation"],
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        save_path,
                        "informer.keras"
                    ),
                    monitor="val_loss",
                    save_best_only=True,
                    initial_value_threshold=np.inf
                ),
            ],
            verbose=0
        )
        print(
            f"Informer training time: {(time.time() - start_time)}", flush=True
        )
    keras_model = keras.saving.load_model(
        os.path.join(save_path, "informer.keras"),
        compile=False
    )
    keras_model.summary()
    results["Informer"] = transfep_utils.benchmark_model(
        keras_model,
        dataset["test"],
        row_wise=True,
        repetitions=REPS
    )

    # Recurrent Seq2Seq
    if not os.path.exists(
            os.path.join(save_path, "recurrent.keras")
    ):
        print("Started RNN training...", flush=True)
        keras_model = transformer.create_recurrent_model(**model_params)
        keras_model.compile(
            keras.optimizers.Adam(model_params["learning_rate"]),
            loss=loss
        )
        start_time = time.time()
        keras_model.fit(
            x=dataset["train"],
            epochs=EPOCHS,
            shuffle=False,
            validation_data=dataset["validation"],
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        save_path,
                        "recurrent.keras"
                    ),
                    monitor="val_loss",
                    save_best_only=True,
                    initial_value_threshold=np.inf
                ),
            ],
            verbose=0
        )
        print(
            f"RNN training time: {(time.time() - start_time)}", flush=True
        )
    keras_model = keras.saving.load_model(
        os.path.join(save_path, "recurrent.keras"),
        compile=False
    )
    keras_model.summary()

    results["RNN"] = transfep_utils.benchmark_model(
        keras_model,
        dataset["test"],
        row_wise=True,
        repetitions=REPS
    )

    # Remove decoder input altogether
    for partition, inputs in dataset.items():
        dataset[partition].x = [
            inputs.x[0],
            inputs.x[1],
            inputs.x[3]
        ]

    # Restore Vanilla Autoformer parameters
    model_params["manual_dec_input"] = False
    model_params["tau"] = min([
        model_params[key]
        for key in model_params
        if key.startswith("tau")
    ])
    model_params["decomposition"] = "ma"

    if not os.path.exists(
            os.path.join(save_path, "autoformer.keras")
    ):
        print("Started Vanilla Autoformer training...", flush=True)
        keras_model = transformer.create_autoformer_model(**model_params)
        keras_model.compile(
            keras.optimizers.Adam(model_params["learning_rate"]),
            loss=loss
        )
        start_time = time.time()
        keras_model.fit(
            x=dataset["train"],
            epochs=EPOCHS,
            shuffle=False,
            validation_data=dataset["validation"],
            callbacks=[
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        save_path,
                        "autoformer.keras"
                    ),
                    monitor="val_loss",
                    save_best_only=True,
                    initial_value_threshold=np.inf
                ),
            ],
            verbose=0
        )
        print(
            f"Autoformer training time: {(time.time() - start_time)}", flush=True
        )
    keras_model = keras.saving.load_model(
        os.path.join(save_path, "autoformer.keras"),
        compile=False
    )
    keras_model.summary()

    results["Autoformer"] = transfep_utils.benchmark_model(
        keras_model,
        dataset["test"],
        row_wise=True,
        repetitions=REPS
    )

    process_results(results)
