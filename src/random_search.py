"""Random-search cross-validation harness, decoupled from dataset and model.

Inject three things and the harness does the rest:
    (a) a dataset  -> arrays passed to :meth:`RandomSearch.run`
    (b) a params table -> the ``param_space`` dict given to the constructor
    (c) a model function -> a :class:`~model_builders.ModelBuilder`

For each randomly sampled config it runs stratified k-fold CV, scores every fold on
accuracy, near-miss and quadratic weighted kappa, and ranks configs by mean CV
accuracy (``avg_vaf``). Swap the ModelBuilder (CNN -> ViT) or the dataset (base ->
GAN) without touching this file.
"""

import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from cnn_base import (
    _CustomEarlyStoppingAndSave,
    _plot_confusion_matrix,
    _plot_epoch_train_val_acc,
    _plot_kfold_val_acc,
    _sum_confusion_matrices,
)
from metrics import near_miss_accuracy, quadratic_weighted_kappa
from model_builders import ModelBuilder


class RandomSearch:
    """Random hyperparameter search with stratified k-fold cross-validation.

    Parameters
    ----------
    model_builder : ModelBuilder
        Produces a compiled model from a sampled params dict.
    param_space : dict
        Maps each hyperparameter to a list of candidate values to sample from.
    k : int
        Number of stratified CV folds (a search setting, not a model hyperparameter).
    seed : int
        Controls both the sampling and the fold shuffling for reproducibility.
    """

    def __init__(self, model_builder: ModelBuilder, param_space: dict, k: int = 5, seed: int = 42):
        self.model_builder = model_builder
        self.param_space = param_space
        self.k = k
        self.seed = seed

    def _cross_validate(self, params: dict, images: np.ndarray, labels: np.ndarray, class_labels):
        """Run stratified k-fold CV for a single config and collect per-fold results."""
        num_classes = len(class_labels)
        input_shape = images.shape[1:]
        epochs = params["epochs"]
        batch_size = params["batch_size"]

        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)

        val_acc_fold, train_acc_fold = [], []
        val_nearmiss_fold, val_qwk_fold = [], []
        val_acc_epoch, train_acc_epoch = [], []
        confusion_matrices = []
        best_model, best_val_acc, best_epoch = None, -1.0, None

        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
            print(f"  Fold {fold + 1}/{self.k}")
            x_train, x_val = images[train_idx], images[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            y_train_enc = tf.keras.utils.to_categorical(y_train, num_classes)
            y_val_enc = tf.keras.utils.to_categorical(y_val, num_classes)

            model = self.model_builder.build(params, input_shape, num_classes)
            early_stop = _CustomEarlyStoppingAndSave(
                patience=4, divergence_threshold=0.15, warmup=15
            )
            history = model.fit(
                x_train,
                y_train_enc,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val_enc),
                verbose=0,
                callbacks=[early_stop],
            )

            _, acc = model.evaluate(x_val, y_val_enc, verbose=0)
            _, tacc = model.evaluate(x_train, y_train_enc, verbose=0)
            val_preds = np.argmax(model.predict(x_val, verbose=0), axis=1)

            val_acc_fold.append(acc)
            train_acc_fold.append(tacc)
            val_nearmiss_fold.append(near_miss_accuracy(y_val, val_preds))
            val_qwk_fold.append(quadratic_weighted_kappa(y_val, val_preds, labels=class_labels))
            val_acc_epoch.append(history.history["val_accuracy"])
            train_acc_epoch.append(history.history["accuracy"])
            confusion_matrices.append({"predicted_values": val_preds, "ground_truth_values": y_val})

            if acc > best_val_acc:
                best_val_acc = acc
                best_model = model
                best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

        return {
            "val_acc_fold": val_acc_fold,
            "train_acc_fold": train_acc_fold,
            "val_nearmiss_fold": val_nearmiss_fold,
            "val_qwk_fold": val_qwk_fold,
            "val_acc_epoch": val_acc_epoch,
            "train_acc_epoch": train_acc_epoch,
            "confusion_matrices": confusion_matrices,
            "best_model": best_model,
            "best_epoch": best_epoch,
        }

    def _test_metrics(self, best_model, test_images, test_labels, class_labels) -> dict:
        """Evaluate the best fold's model on a held-out test set."""
        test_labels = np.asarray(test_labels).astype(int)
        test_enc = tf.keras.utils.to_categorical(test_labels, len(class_labels))
        _, test_acc = best_model.evaluate(test_images, test_enc, verbose=0)
        test_preds = np.argmax(best_model.predict(test_images, verbose=0), axis=1)
        return {
            "test_acc": float(test_acc),
            "test_nearmiss": near_miss_accuracy(test_labels, test_preds),
            "test_qwk": quadratic_weighted_kappa(test_labels, test_preds, labels=class_labels),
        }

    def _save_artifacts(self, folder, cv, class_labels, index) -> None:
        """Write the plots and the best-fold Keras model for one config into ``folder``."""
        os.makedirs(folder, exist_ok=True)
        prefix = folder + os.sep
        _plot_epoch_train_val_acc(prefix, cv["train_acc_epoch"], cv["val_acc_epoch"], self.k)
        _plot_kfold_val_acc(prefix, self.k, cv["val_acc_fold"])
        summed_cm = _sum_confusion_matrices(class_labels, cv["confusion_matrices"])
        _plot_confusion_matrix(
            summed_cm,
            np.array(class_labels),
            ("Predicted Label", "Ground Truth Label"),
            os.path.join(folder, "confusion_matrix.png"),
        )
        if cv["best_model"] is not None:
            cv["best_model"].save(os.path.join(folder, f"model_{index}.keras"))

    def _update_top_n(self, saved: list, top_n: int, avg_vaf, index, cv, class_labels, output_dir):
        """Keep artefacts only for the current top ``top_n`` models by avg_vaf.

        Saves this model's plots+model iff it qualifies, evicting and deleting the
        worst-kept model's folder when full. Peak disk usage is ``top_n`` folders --
        nothing is created then cropped afterwards.
        """
        if len(saved) >= top_n:
            worst = min(saved, key=lambda s: s["avg_vaf"])
            if avg_vaf <= worst["avg_vaf"]:
                return  # not good enough to keep -> never write its artefacts
            shutil.rmtree(worst["folder"], ignore_errors=True)
            saved.remove(worst)
        folder = os.path.join(output_dir, f"model{index}_{avg_vaf * 100:.2f}")
        self._save_artifacts(folder, cv, class_labels, index)
        saved.append({"avg_vaf": avg_vaf, "folder": folder})

    def _sample_valid_configs(self, n_models: int, input_shape: tuple) -> list:
        """Draw ``n_models`` distinct configs the model builder can actually build.

        Random search with rejection: sample a value per hyperparameter, skip duplicates
        and any config the builder rejects (``model_builder.is_valid``), until ``n_models``
        are collected. Reproducible for a given seed. If the valid space is smaller than
        ``n_models`` it returns everything valid and warns.
        """
        rng = random.Random(self.seed)
        keys = list(self.param_space)
        configs, seen = [], set()
        # Generous attempt cap so an impossible request terminates instead of looping.
        max_attempts = max(n_models * 500, 10_000)
        for _ in range(max_attempts):
            if len(configs) >= n_models:
                break
            params = {key: rng.choice(self.param_space[key]) for key in keys}
            signature = tuple(sorted(params.items()))
            if signature in seen:
                continue
            seen.add(signature)
            if self.model_builder.is_valid(params, input_shape):
                configs.append(params)
        if len(configs) < n_models:
            print(
                f"WARNING: only {len(configs)} valid unique configs found "
                f"(requested {n_models}); the valid parameter space may be small."
            )
        return configs

    def run(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        test_images: np.ndarray = None,
        test_labels: np.ndarray = None,
        n_models: int = 4,
        output_dir: str = ".",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """Sample ``n_models`` valid configs, cross-validate each, and rank by mean CV accuracy.

        Storage-efficient by design (for large searches on little disk):
          - EVERY model's metrics (incl. training time) stream into a single
            ``random_search_results.csv`` as they finish -- one row per model, no
            per-model CSVs.
          - Plots and the Keras model are kept only for the current top ``top_n``
            models by ``avg_vaf``. This is tracked incrementally: a model's artefacts
            are written only if it qualifies, and the worst-kept model's folder is
            deleted when a better one arrives -- so at most ``top_n`` folders ever exist
            (nothing is created then cropped afterwards).

        Returns the leaderboard DataFrame (sorted by ``avg_vaf`` descending).
        """
        train_labels = np.asarray(train_labels).astype(int)
        class_labels = sorted(np.unique(train_labels).tolist())

        sampled = self._sample_valid_configs(n_models, train_images.shape[1:])
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "random_search_results.csv")
        if os.path.exists(results_path):
            os.remove(results_path)  # start a fresh streamed results file
        print(f"Sampled {len(sampled)} valid configs; keeping artefacts for the top {top_n}")

        rows, saved, header_written = [], [], False
        for index, params in enumerate(sampled):
            print(f"\n=== Model {index + 1}/{len(sampled)} ===\n{params}")
            start = time.perf_counter()
            try:
                cv = self._cross_validate(params, train_images, train_labels, class_labels)
            except Exception as exc:  # defensive: a runtime failure shouldn't abort the sweep
                print(f"Model {index} failed to build/train ({exc}); skipping.")
                continue
            train_seconds = time.perf_counter() - start

            avg_vaf = float(np.mean(cv["val_acc_fold"]))
            avg_taf = float(np.mean(cv["train_acc_fold"]))
            row = dict(params)
            row.update(
                {
                    "model_id": index,
                    "avg_vaf": avg_vaf,
                    "avg_taf": avg_taf,
                    "avg_divergence": avg_taf - avg_vaf,
                    "avg_val_nearmiss": float(np.mean(cv["val_nearmiss_fold"])),
                    "avg_val_qwk": float(np.mean(cv["val_qwk_fold"])),
                    "best_epoch": cv["best_epoch"],
                    "train_seconds": round(train_seconds, 1),
                }
            )
            if test_images is not None and test_labels is not None and cv["best_model"] is not None:
                row.update(
                    self._test_metrics(cv["best_model"], test_images, test_labels, class_labels)
                )
            rows.append(row)

            # Stream this row to disk immediately (crash-safe, constant memory).
            pd.DataFrame([row]).to_csv(
                results_path, mode="a", header=not header_written, index=False
            )
            header_written = True

            # Save plots + model only while this config is in the top-N.
            self._update_top_n(saved, top_n, avg_vaf, index, cv, class_labels, output_dir)

        if not rows:
            raise RuntimeError("No configs completed successfully.")

        # Rewrite the results file once at the end, sorted by mean CV accuracy.
        leaderboard = (
            pd.DataFrame(rows).sort_values("avg_vaf", ascending=False).reset_index(drop=True)
        )
        leaderboard.to_csv(results_path, index=False)
        print(f"\n{len(rows)} models -> {results_path}; artefacts kept for top {len(saved)}.")
        return leaderboard
