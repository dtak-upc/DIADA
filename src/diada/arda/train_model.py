import pandas as pd
import numpy as np
import tempfile
import shutil

from autogluon.tabular import TabularPredictor
from autogluon.core import TabularDataset

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastai")
import logging
logger = logging.getLogger(__name__)

import random
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

class ModelTraining:
    def __init__(self, dataset: pd.DataFrame, target: str, time_limit: int = 120, presets: str = "high_quality", task: str = "classification", eval_metric: str = "accuracy"):
        self.df = dataset.copy()
        self.target = target
        if target not in self.df.columns:
            raise ValueError(f"Target column '{target}' not found in the df.")

        self.time_limit = time_limit
        self.task = task.strip().lower()
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be either 'classification' or 'regression'.")
        
        if self.task == "classification":
            self.problem_type = "multiclass" if self.df[target].nunique() > 2 else "binary"
            self.average = "macro"
            if eval_metric not in ("accuracy", "f1"):
                logger.error("eval_metric must be either 'accuracy' or 'f1'. Defaulting to 'accuracy'")
                eval_metric = "accuracy"
        else:
            self.problem_type = "regression"
            if eval_metric not in ("root_mean_squared_error", "mae"):
                logger.error("eval_metric must be either 'root_mean_squared_error' or 'mae'.Defaulting to 'root_mean_squared_error'")
                eval_metric = "root_mean_squared_error"
        
        self.stratify = self.df[target] if self.task == "classification" else None
        self.eval_metric = eval_metric
        self.presets = presets

        self.leaderboard = None
        self.metrics = None
        self.predictor = None

        self.tmp_dir = tempfile.mkdtemp() # Temporal directory to store the models and delete them

    def train_model_base(self):
        try:
            train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=SEED, stratify=self.stratify)

            train_data = TabularDataset(train_df)
            test_data  = TabularDataset(test_df)
            X_test = test_data.drop(columns=[self.target])
            y_test = test_data[self.target]

            self.predictor = TabularPredictor(label=self.target, problem_type=self.problem_type, 
                                              eval_metric=self.eval_metric, verbosity=0, path=self.tmp_dir, learner_kwargs={"random_state": SEED})
            self.predictor = self.predictor.fit(train_data=train_data, time_limit=self.time_limit, presets=self.presets, keep_only_best=True, 
                                                num_bag_folds=0, num_stack_levels=0, ag_args_fit={"random_seed": SEED}, included_model_types=["GBM", "XGB", "RF", "XT"])
            y_pred = self.predictor.predict(X_test) # Evaluation on the held-out test set

            # Training predictions (to measure overfit)
            X_train = train_data.drop(columns=[self.target])
            y_train = train_data[self.target]
            y_pred_train = self.predictor.predict(X_train)

            self.leaderboard = self.predictor.leaderboard(test_data, silent=True) #  Leaderboard of all trained models

            if self.task == "classification":
                y_prob = self.predictor.predict_proba(X_test) # Probabilities for AUC

                if self.problem_type == "binary":
                    # predict_proba returns a DataFrame with one column per class; we need the positive-class column (last one).
                    auc = roc_auc_score(y_test, y_prob.iloc[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
                
                test_accuracy = accuracy_score(y_test, y_pred)
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_f1 = f1_score(y_test, y_pred, average=self.average, zero_division=0)
                train_f1 = f1_score(y_train, y_pred_train, average=self.average, zero_division=0)

                self.metrics = {
                    "test_accuracy": test_accuracy,
                    "test_f1": f1_score(y_test, y_pred, average=self.average, zero_division=0),
                    # "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "test_precision" : precision_score(y_test, y_pred, average=self.average, zero_division=0),
                    "test_recall": recall_score(y_test, y_pred, average=self.average, zero_division=0),
                    "test_auc": auc,
                    "overfit_gap_accuracy": test_accuracy - train_accuracy,
                    "overfit_gap_f1": test_f1 - train_f1,
                }

            else:  # regression
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_mae = mean_absolute_error(y_test, y_pred)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred)

                self.metrics = {
                    "test_rmse": test_rmse, 
                    "test_mae": test_mae,
                    "test_r2": test_r2,
                    "overfit_rmse": test_rmse - train_rmse, 
                    "overfit_mae": test_mae - train_mae,
                }

            return self.metrics
        finally:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
    
    def print_results(self):
        if self.leaderboard is None:
            logger.error("No models were trained, so there is not anything to show")
        else: # Summary print
            print("\n" + "=" * 60)
            print(f"  AutoGluon – {self.task.capitalize()} Results")
            print("=" * 60)
            print(f"  Best model : {self.predictor.model_best}")
            print(f"  Task       : {self.problem_type}")
            print(f"  Eval metric: {self.eval_metric}\n")
            print("  Test-set metrics:")
            for name, value in self.metrics.items():
                print(f"    {name:<12}: {value:.4f}")
            print("=" * 60 + "\n")
            print("  Model leaderboard (top 5):")
            print(self.leaderboard.head(5).to_string(index=False))
            print("=" * 60 + "\n")