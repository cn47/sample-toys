import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from pprint import pformat

import hydra
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from logger import get_logger
from mlflow_writer import MlflowWriter
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split
)
from utils import rm_files, timer

pj_dir = Path("/opt")
config = OmegaConf.load(f"{pj_dir}/src/config/config.yaml")

mlflow_dir = pj_dir / config.mlflow.dir
hydra_dir = pj_dir / f"data/hydra/{datetime.now():%Y-%m-%d}/{datetime.now():%H-%M-%S}"
sys.argv.append(f"hydra.run.dir={hydra_dir}")
sys.argv.append(f"hydra.sweep.dir={hydra_dir}")

logger = get_logger("TrainOptimizer", f"{pj_dir}/log/train.log")


### Define Process #############################################################
def main():
    with timer("Load&Preprocess Data"):
        global X, y
        df_raw = pd.read_csv(pj_dir / "data/01_raw/train.csv")
        df_proc = preprocess(df_raw)
        X, y = df_proc.drop("Survived", axis=1), df_proc["Survived"]
        positive = np.count_nonzero(y)
        negative = len(y) - np.count_nonzero(y)
        logger.info(f"positive: {positive} / negative: {negative}")

    with timer("CrossValidHyperParamOptimizer"):
        optimizer()

    with timer("TrainBestModel", logger):
        train_by_best_params()


### Define Function ############################################################
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    _df["FamilySize"] = _df["SibSp"] + _df["Parch"] + 1
    _df["Embarked"] = _df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    _df["Sex"] = _df["Sex"].map({"male": 0, "female": 1})
    _df["IsAlone"] = 0
    _df.loc[_df["FamilySize"] == 1, "IsAlone"] = 1
    _df.drop(
        ["PassengerId", "Name", "Age", "SibSp", "Parch", "Fare", "Cabin", "Ticket"],
        axis=1,
        inplace=True,
    )

    return _df


@hydra.main(config_path=f"{pj_dir}/src/config", config_name="config")
def optimizer(config: DictConfig) -> np.float64:
    cv = StratifiedKFold(
        n_splits=config.common.fold,
        shuffle=True,
        random_state=config.common.seed
    )

    model = LGBMClassifier(random_state=config.common.seed)
    hyperparams = {**config.model.train_params}
    model.set_params(**hyperparams)

    fit_params = {
        "verbose": 0,
        "early_stopping_rounds": config.model.callbacks.early_stopping_rounds,
        "eval_metric": config.model.callbacks.metric,
        "eval_set": [(X, y)],
    }

    with timer("CrossValidScore", logger):
        scores = cross_validate(
            model, X, y,
            scoring=list(config.metrics), cv=cv,
            fit_params=fit_params, n_jobs=-1
        )

    logger.info("---- CrossValid Scores\n")
    logger.info(pformat(scores))

    writer = MlflowWriter(
        experiment_name=config.mlflow.experiment_name,
        tracking_uri=f"file://{mlflow_dir}/mlruns",
    )

    current_dir = Path().absolute()

    if current_dir.stem.startswith("trial_"):
        trial_num = int(current_dir.stem.split("_")[1])
        mlflow_run_name = f"SweepTrial{trial_num:03}"
    else:
        mlflow_run_name = "ShotTrial"

    tags = {
        "RunAt": f"{datetime.now():%Y-%m-%d-%H-%M-%S}",
        "mlflow.runName": mlflow_run_name,
    }

    if config.mlflow.get("tags"):
        tags.update(config.mlflow.tags)

    writer.set_tags(tags)

    writer.log_params_from_omegaconf_dict(hyperparams)

    mean_scores = {f"mean_{k}".replace("_test", ""): v.mean() for k, v in scores.items()}
    std_scores = {f"std_{k}".replace("_test", ""): v.std() for k, v in scores.items()}

    [writer.log_metric(k, v) for k, v in mean_scores.items()]
    [writer.log_metric(k, v) for k, v in std_scores.items()]

    writer.log_artifact(current_dir / ".hydra" / "config.yaml")
    writer.log_artifact(current_dir / ".hydra" / "hydra.yaml")
    writer.log_artifact(current_dir / ".hydra" / "overrides.yaml")
    writer.log_artifact(current_dir / f"{Path(os.path.basename(__file__)).stem}.log")

    writer.set_terminated()

    return np.mean(scores["test_average_precision"])


def train_by_best_params() -> None:

    rm_files(mlflow_dir / "_tmp")
    (mlflow_dir / "_tmp").mkdir(parents=True, exist_ok=True)

    hyperparams = {**config.model.train_params}

    optimization_results_file = hydra_dir / "optimization_results.yaml"
    if optimization_results_file.exists():
        searched_params = OmegaConf.load(optimization_results_file)
        searched_params = {
            k.replace("model.train_params.", ""): v for k, v in searched_params.best_params.items()
        }
        hyperparams.update(searched_params)

    logger.info("fit_params:")
    logger.info(pformat(hyperparams))

    (X_train, X_valid, y_train, y_valid,) = train_test_split(
        X,
        y,
        test_size=config.common.test_size,
        random_state=config.common.seed,
        shuffle=True,
        stratify=y,
    )

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)

    callbacks = [
        lgb.early_stopping(
            stopping_rounds=config.model.callbacks.early_stopping_rounds, verbose=True
        ),
        lgb.log_evaluation(10),
    ]

    evals_result = {}

    model = lgb.train(
        hyperparams,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["Train", "Eval"],
        evals_result=evals_result,
        callbacks=callbacks,
    )

    logger.info("Train Done")

    plt.style.use("seaborn-pastel")
    fig = plt.figure()

    ax = lgb.plot_metric(evals_result, grid=False)
    plt.savefig(f"{mlflow_dir}/_tmp/LearningCurve.png", bbox_inches="tight", pad_inches=0.05)

    ax = lgb.plot_importance(model, max_num_features=5, grid=False)
    plt.savefig(f"{mlflow_dir}/_tmp/FeatureImportance5.png", bbox_inches="tight", pad_inches=0.05)

    with open(f"{mlflow_dir}/_tmp/lgbm_optimized.pkl", "wb") as fp:
        pickle.dump(model, fp)

    writer = MlflowWriter(
        experiment_name=config.mlflow.experiment_name,
        tracking_uri=f"file://{mlflow_dir}/mlruns",
    )

    tags = {
        "RunAt": f"{datetime.now():%Y-%m-%d-%H-%M-%S}",
        "mlflow.runName": "TrainOptimizedModel",
    }

    if config.mlflow.get("tags"):
        tags.update(config.mlflow.tags)

    writer.set_tags(tags)

    writer.log_params_from_omegaconf_dict(hyperparams)

    writer.log_param("best_iteration", model.best_iteration)
    writer.log_param("X_train_shape", str(X_train.shape))
    writer.log_param("y_train_shape", round(np.mean(y_train), 3))
    writer.log_param("X_valid_shape", str(X_valid.shape))
    writer.log_param("y_valid_shape", round(np.mean(y_valid), 3))

    writer.log_metric("BestScore_train", list(model.best_score["Train"].values())[0])
    writer.log_metric("BestScore_valid", list(model.best_score["Eval"].values())[0])

    writer.log_artifact(f"{mlflow_dir}/_tmp/lgbm_optimized.pkl")
    writer.log_artifact(f"{mlflow_dir}/_tmp/LearningCurve.png")
    writer.log_artifact(f"{mlflow_dir}/_tmp/FeatureImportance5.png")

    writer.set_terminated()


### Execute Process ############################################################
if __name__ == "__main__":
    main()
