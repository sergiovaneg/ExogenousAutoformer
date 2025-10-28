"""
Module with custom optuna utilities
"""

import os
import copy

import numpy as np
import keras
from keras import KerasTensor as Tensor
from keras import ops
import jax

from jax.errors import JaxRuntimeError

import optuna

from typing import Callable, Optional

EPS = 1e-9

type GenericLoss = keras.Loss | Callable[[Tensor, Tensor], float] | str
type GenericMetric = keras.Metric | Callable[[Tensor, Tensor], float] | str


def symmetric_mean_average_percentual_error(
    y_true: Tensor, y_pred: Tensor
):
    num = ops.abs(y_true - y_pred)

    den = (ops.abs(y_true) + ops.abs(y_pred)) / 2
    den = ops.where(den > 0, den, EPS)

    return ops.mean(ops.divide(num, den), axis=-1)


def signed_log(x: Tensor) -> Tensor:
    return ops.sign(x) * ops.log1p(ops.abs(x))


def mean_squared_signed_logarithmic_error(
    y_true: Tensor, y_pred: Tensor
):
    return ops.mean(ops.square(signed_log(
        y_true) - signed_log(y_pred)), axis=-1)


def symmetric_normalized_mean_squared_error(
    y_true: Tensor, y_pred: Tensor
):
    num = ops.square(y_true - y_pred)

    den = (ops.square(y_true) + ops.square(y_pred))
    den = ops.where(den > 0, den, EPS)

    return ops.mean(ops.divide(num, den), axis=-1)


def mean_arctangent_squared_percentual_error(
    y_true: Tensor, y_pred: Tensor
):
    return ops.mean(
        ops.arctan(
            ops.square(
                ops.divide(
                    y_true - y_pred,
                    ops.where(
                        ops.abs(y_true) < 1e-7,
                        1e-7,
                        y_true
                    )
                )
            )
        ),
        axis=-1
    )


class KerasValidationPruner(keras.callbacks.Callback):
    """
    Keras callback to prune trial during training
    """

    def __init__(
        self,
        trial: optuna.Trial,
        metrics: tuple[GenericMetric, ...]
    ):
        super().__init__()
        self.trial = trial
        self.metrics = metrics

    def on_epoch_end(self, epoch: int, logs: dict = None):
        loss = float(logs.get("val_loss", logs["loss"]))
        self.trial.report(loss, step=epoch)

        if self.trial.should_prune():
            for metric in self.metrics:
                if not isinstance(metric, str):
                    metric = metric.__name__
                self.trial.set_user_attr(
                    metric,
                    logs["val_" + metric]
                )

            raise optuna.TrialPruned()


def parse_key(
    trial: optuna.Trial,
    param_type: str,
    key: str, key_params: dict
) -> int | float:
    match param_type:
        case "int":
            sampler = trial.suggest_int
        case "discrete_uniform":
            sampler = trial.suggest_discrete_uniform
        case "float":
            sampler = trial.suggest_float
        case "uniform":
            sampler = trial.suggest_uniform
        case "categorical":
            sampler = trial.suggest_categorical
    return sampler(key, **key_params)


class CleanupCallback:
    """
    Callback to delete unneeded trials
    """

    def __init__(
        self,
        save_path: str
    ):
        self.save_path = save_path

    def __call__(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial
    ):
        tmp = os.path.join(self.save_path, "tmp.keras")
        if not os.path.exists(tmp):
            return

        try:
            best = study.best_trial
        except ValueError:
            os.remove(tmp)
            return

        if best.number == trial.number:
            os.rename(
                tmp,
                os.path.join(self.save_path, "best.keras")
            )
        else:
            os.remove(tmp)


class StoreBestCallback:
    """
    Outputs the current best results to an npz file
    """

    def __init__(
        self, *,
        save_path: str,
        raw_data: any,
        formatter: Callable[..., dict],
        model_params: Optional[dict] = None,
    ):
        self.save_path = save_path
        self.raw_data = raw_data
        self.formatter = formatter
        self.model_params = model_params or {}

        os.makedirs(
            save_path,
            exist_ok=True,
            mode=0o774
        )

    def __call__(
        self,
        study_: optuna.study.Study,
        trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE \
                and study_.best_trial.number == trial.number:
            model_params = copy.deepcopy(self.model_params)
            model_params.update(trial.params)

            model_params["batch_size"] = model_params.get("batch_size", 64)
            model_params["validation_batch_size"] = max(
                1,
                round(np.log(model_params["batch_size"]))
            )

            dataset = self.formatter(self.raw_data, model_params)
            model = keras.saving.load_model(
                os.path.join(
                    self.save_path,
                    f"model_{trial.number}.keras"
                ),
                compile=False
            )

            file_descriptor = os.open(path=os.path.join(
                self.save_path,
                f"model_{trial.number}.npz"),
                flags=os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                mode=0o774)
            with open(file_descriptor, "w+b") as file:
                try:
                    y_hat = model.predict(
                        dataset["test"],
                        batch_size=model_params["validation_batch_size"],
                        verbose=2
                    )
                except JaxRuntimeError:
                    y_hat = np.full_like(dataset["test"].y, np.nan)

                np.savez_compressed(
                    file,
                    y=dataset["test"].y,
                    y_hat=y_hat,
                    **dict([
                        (f"x_{idx}", e)
                        for idx, e in enumerate(dataset["test"].x)
                    ])
                )


class Seq2SeqObjective:
    """
    Generic optuna objective wrapper class for keras optimization
    """

    def __init__(
        self, *,
        save_path: str,
        search_params: dict,
        raw_data: any,
        formatter: Callable[..., dict],
        model_factory: Callable[..., keras.Model],
        loss: GenericLoss,
        metrics: Optional[tuple[GenericMetric, ...]] = None,
        model_params: Optional[dict] = None,
        model_checker: Optional[Callable[[dict], bool]] = None
    ) -> None:
        self.save_path = save_path
        self.search_params = search_params
        self.raw_data = raw_data
        self.formatter = formatter
        self.model_factory = model_factory
        self.loss = loss
        self.metrics = metrics
        self.model_params = model_params or {}
        self.model_checker = model_checker or (lambda _: True)

        os.makedirs(
            save_path,
            exist_ok=True,
            mode=0o774
        )

    def __call__(self, trial: optuna.Trial) -> float:
        model_params = copy.deepcopy(self.model_params)
        search_params = copy.deepcopy(self.search_params)
        for metric in search_params:
            if metric == "tau":
                continue
            param_type = search_params[metric].pop("type")
            model_params[metric] = parse_key(
                trial, param_type, metric,
                search_params[metric]
            )

        # Exception for multiple tau
        if "tau" in search_params:
            model_params["tau"] = []
            tau_type = search_params["tau"].pop("type")
            search_params["tau"]["high"] = min(model_params["I"],
                                               search_params["tau"]["high"])
            n_blocks = model_params.get("n_blocks", None) or \
                max(model_params.get("N"), model_params.get("M"))
            for idx in range(n_blocks):
                model_params["tau"].append(
                    parse_key(trial, tau_type,
                              "tau_" + str(idx),
                              search_params["tau"])
                )
                search_params["tau"]["high"] = min(search_params["tau"]["high"],
                                                   model_params["tau"][-1])

        if not self.model_checker(model_params):
            raise optuna.TrialPruned()

        model_params["batch_size"] = model_params.get("batch_size", 64)
        model_params["validation_batch_size"] = max(
            1,
            round(np.log(model_params["batch_size"]))
        )

        dataset = self.formatter(self.raw_data, model_params)
        learning_rate = model_params.get("learning_rate", 1e-3)
        model = self.model_factory(**model_params)

        trial.set_user_attr("Weight count", model.count_params())

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=self.loss,
            metrics=self.metrics
        )

        epochs = model_params.get("epochs", 100)

        try:
            history = model.fit(
                x=dataset["train"],
                batch_size=model_params["batch_size"],
                epochs=epochs,
                shuffle=False,
                validation_data=dataset["validation"],
                validation_batch_size=model_params["validation_batch_size"],
                callbacks=[
                    keras.callbacks.TerminateOnNaN(),
                    keras.callbacks.ModelCheckpoint(
                        os.path.join(
                            self.save_path,
                            "tmp.keras"
                        ),
                        monitor="val_loss",
                        save_best_only=True,
                        initial_value_threshold=np.inf
                    ),
                    KerasValidationPruner(trial, self.metrics)
                ],
                verbose=0
            )
        except JaxRuntimeError as e:
            raise optuna.TrialPruned() from e

        jax.clear_caches()

        loss_idx = history.history["val_loss"].index(
            min(history.history["val_loss"]))
        loss = history.history["val_loss"][loss_idx]
        for metric in history.history:
            if metric == "val_loss" or not metric.startswith("val_"):
                continue
            trial.set_user_attr(
                metric.removeprefix("val_"),
                history.history[metric][loss_idx]
            )

        return loss
