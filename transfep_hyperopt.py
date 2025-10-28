"""
Script to optimize the hyperparameters of the Autoformer++ for energy
prosumption models
"""

import sys
import json
from functools import partial

import optuna
import keras

from keras_transformer import generic_transformer as transformer

import optuna_utils
import transfep_utils

N_TRIALS = 250
EPOCHS = 1000


args = transfep_utils.get_cli_args()

ROOT = args.root
DATABASE = "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/transfep"

STUDY_FILEPATH: str = args.study

BATCH_SIZE: int = args.batch_size

Y: str = args.target

I: int = args.context_window
O: int = args.prediction_horizon
STRIDE: int = 1  # Hourly stride

keras.utils.set_random_seed(0)

with open(
    "./params/variables.json",
    "r",
    encoding=sys.getdefaultencoding()
) as f:
    VAR_JSON = json.load(f)


def transformer_checker(params: dict) -> bool:
    if params["h"] > params["d_model"]:
        return False
    return True


if __name__ == "__main__":
    relevant_variables = VAR_JSON[Y]
    loss, study_name, save_path = transfep_utils.get_loss_and_paths(args)

    raw_data = transfep_utils.get_curated_datasets(
        relevant_variables + [Y],
        # Using logarithmic loss instead of modifying the signal
        log_solar=False,
        root=ROOT
    )

    with open(
        f"./params/{STUDY_FILEPATH}.json",
        "r",
        encoding=sys.getdefaultencoding()
    ) as f:
        search_params = json.load(f)

    model_params = {
        "I": I,
        "O": O,
        "manual_dec_input": True,
        "normalize": False,
        "d_enc": len(relevant_variables) + 1,
        "d_dec": len(relevant_variables) + 1,
        "d_out": 1,
        "output_components": False,
        "output_attention": False,
        "embed_type": "temporal",
        "freq": "h",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-4
    }

    formatter = partial(
        transfep_utils.dataset_formatter,
        target=Y,
        updatable=False,
        iterative=False,  # Iterative w/ Transfer Learning
        update_stride=48,
        normalize=Y == "full_solar",  # Normalize only for Production
        context_days=30 if Y == "full_solar" else 0,
        sigma=0.001,  # .1% of variable's delta std at 1h
        beta=0.03  # 1% of variable's delta std at 72h
    )

    objective = optuna_utils.Seq2SeqObjective(
        save_path=save_path,
        search_params=search_params,
        raw_data=raw_data,
        formatter=partial(formatter, stride=STRIDE),
        model_factory=transformer.create_autoformer_model,
        loss=loss,
        metrics=[keras.metrics.mean_absolute_error],
        model_params=model_params,
        model_checker=transformer_checker
    )

    cleanup_callback = optuna_utils.CleanupCallback(
        save_path=save_path
    )

    study = optuna.create_study(
        storage=DATABASE,
        direction="minimize",
        load_if_exists=True,
        study_name=study_name,
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=round(EPOCHS / 10)
        ),
        sampler=optuna.samplers.TPESampler(seed=0)
    )

    for past_trial in study.trials:
        if past_trial.state == optuna.trial.TrialState.RUNNING:
            study.enqueue_trial(past_trial.params)
            study.tell(past_trial.number, state=optuna.trial.TrialState.FAIL)

    study.optimize(
        func=objective,
        n_trials=N_TRIALS,
        callbacks=[cleanup_callback,]
    )
