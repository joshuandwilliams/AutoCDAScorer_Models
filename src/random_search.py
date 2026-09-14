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

import gc
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold, StratifiedKFold

from cnn_base import (
    _CustomEarlyStoppingAndSave,
    _plot_confusion_matrix,
    _plot_epoch_train_val_acc,
    _plot_kfold_val_acc,
    _sum_confusion_matrices,
)
from metrics import near_miss_accuracy, quadratic_weighted_kappa
from model_builders import ModelBuilder
from results_store import JobCounter, TopNStore, append_rows_to_csv


def _rss_mb():
    """This process's resident set size in MB, or None where unavailable.

    Reads /proc/self/status, so it returns None off Linux (macOS has no procfs). The
    callers treat None as "no ceiling", which is the safe default: the HPC is Linux.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


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
        Seeds the config *sampling* only -- vary it per task (e.g. the SLURM array
        index) so different tasks explore different configs.
    cv_seed : int
        Seeds the *CV fold split*. Keep it the SAME across all tasks so every model in
        the whole search is scored on the identical folds -- a fair global ranking.
    run_id : str
        Identifier for this task (e.g. the SLURM array index). Prefixes model ids so
        they stay globally unique when many tasks share one output folder.
    """

    def __init__(
        self,
        model_builder: ModelBuilder,
        param_space: dict,
        k: int = 5,
        seed: int = 42,
        cv_seed: int = 42,
        run_id: str = "0",
    ):
        self.model_builder = model_builder
        self.param_space = param_space
        self.k = k
        self.seed = seed
        self.cv_seed = cv_seed
        self.run_id = str(run_id)

    def _folds(
        self, images: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None
    ) -> list:
        """Stratified k-fold split. Uses ``cv_seed`` (shared across tasks), not the
        per-task sampling ``seed``, so all models are compared on the same folds.

        ``groups`` keeps rows that share a group in the same fold. For the augmented runs
        the group is the index of the real crop an augmented copy was made from, so a crop
        and its augmentations are never split across the training and validation sides.
        Without that, a validation fold would hold augmented copies of images whose
        originals were trained on, and the validation accuracy would measure recall of the
        training set rather than generalisation.
        """
        if groups is None:
            skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.cv_seed)
            return list(skf.split(images, labels))
        sgk = StratifiedGroupKFold(n_splits=self.k, shuffle=True, random_state=self.cv_seed)
        return list(sgk.split(images, labels, groups=groups))

    def _cross_validate(
        self,
        params: dict,
        images: np.ndarray,
        labels: np.ndarray,
        class_labels,
        soft_targets: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        is_aug: np.ndarray | None = None,
    ):
        """Run stratified k-fold CV for a single config and collect per-fold results.

        ``soft_targets`` optionally replaces the one-hot TRAINING target with a full
        probability vector per image -- for this project, the distribution of the three
        scorers' scores rather than a one-hot of their median. Validation is deliberately
        left on the hard median label, so ``avg_vaf`` keeps the same meaning as in every
        other search and the runs stay comparable. Fold assignment also still uses the
        hard label, so the stratification is unchanged.

        Memory is freed per FOLD, not just per config: each fold's model graph is released
        (``clear_session`` + ``gc``) before the next fold builds, so only ONE model is ever
        alive at a time. The previous code cleared only once per config and held the best
        fold's live model object, so all k folds' graphs accumulated within a single config
        -- fine for the tiny CNNs but enough to exhaust GPU memory for the much larger ViTs.
        The best fold is instead kept as its *weights* (detached numpy arrays that survive
        ``clear_session``); its model is rebuilt once at the end for saving and test scoring.
        """
        tf.keras.backend.clear_session()
        gc.collect()

        num_classes = len(class_labels)
        input_shape = images.shape[1:]
        epochs = params["epochs"]
        batch_size = params["batch_size"]

        val_acc_fold, train_acc_fold = [], []
        # Accuracy split by whether the crop is a real photograph or an augmented copy.
        # A model that scores far higher on the augmented half has learned something about
        # the augmentation rather than about severity, which the pooled number would hide.
        val_acc_orig_fold, val_acc_aug_fold = [], []
        train_acc_orig_fold, train_acc_aug_fold = [], []
        val_nearmiss_fold, val_qwk_fold = [], []
        val_acc_epoch, train_acc_epoch = [], []
        confusion_matrices = []
        best_weights, best_val_acc, best_epoch = None, -1.0, None

        for fold, (train_idx, val_idx) in enumerate(self._folds(images, labels, groups)):
            print(f"  Fold {fold + 1}/{self.k}")
            x_train, x_val = images[train_idx], images[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            if soft_targets is not None:
                y_train_enc = soft_targets[train_idx]
            else:
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

            val_preds = np.argmax(model.predict(x_val, verbose=0), axis=1)
            # Train accuracy is scored against the HARD label even when training on soft
            # targets. Categorical accuracy compares argmax(y_true) to argmax(y_pred), and
            # the argmax of a soft target is the modal scorer's score, which is not always
            # the median. Using the hard label keeps avg_taf and avg_divergence directly
            # comparable with the one-hot runs.
            train_preds = np.argmax(model.predict(x_train, verbose=0), axis=1)
            acc = float(np.mean(val_preds == y_val))
            tacc = float(np.mean(train_preds == y_train))

            def _split(correct, mask):
                """Accuracy over a subset; nan when the subset is empty (e.g. class 0 in
                the class-0-matched run receives no augmented crops at all)."""
                return float(np.mean(correct[mask])) if mask.any() else float("nan")

            if is_aug is not None:
                va, ta = is_aug[val_idx], is_aug[train_idx]
                val_acc_orig_fold.append(_split(val_preds == y_val, ~va))
                val_acc_aug_fold.append(_split(val_preds == y_val, va))
                train_acc_orig_fold.append(_split(train_preds == y_train, ~ta))
                train_acc_aug_fold.append(_split(train_preds == y_train, ta))

            val_acc_fold.append(acc)
            train_acc_fold.append(tacc)
            val_nearmiss_fold.append(near_miss_accuracy(y_val, val_preds))
            val_qwk_fold.append(quadratic_weighted_kappa(y_val, val_preds, labels=class_labels))
            # Copy the per-epoch history out (plain lists) so the History object can be freed.
            val_acc_epoch.append(list(history.history["val_accuracy"]))
            train_acc_epoch.append(list(history.history["accuracy"]))
            confusion_matrices.append({"predicted_values": val_preds, "ground_truth_values": y_val})

            if acc > best_val_acc:
                # Keep the best fold's WEIGHTS (numpy copies), not the live model -- they
                # survive the clear_session below, so nothing is held across folds.
                best_val_acc = acc
                best_weights = model.get_weights()
                best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

            # Release this fold's graph/GPU memory before the next fold's model is built.
            del model, history, early_stop
            tf.keras.backend.clear_session()
            gc.collect()

        # Rebuild the best fold's model once, from its saved weights, for artefacts + test.
        best_model = None
        if best_weights is not None:
            best_model = self.model_builder.build(params, input_shape, num_classes)
            best_model.set_weights(best_weights)

        return {
            "val_acc_fold": val_acc_fold,
            "train_acc_fold": train_acc_fold,
            "val_acc_orig_fold": val_acc_orig_fold,
            "val_acc_aug_fold": val_acc_aug_fold,
            "train_acc_orig_fold": train_acc_orig_fold,
            "train_acc_aug_fold": train_acc_aug_fold,
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

    def _save_artifacts(self, folder, cv, class_labels) -> None:
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
            cv["best_model"].save(os.path.join(folder, "model.keras"))

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

    def all_valid_configs(self, input_shape: tuple) -> list:
        """Every buildable config in the full grid, in a fixed deterministic order.

        All self-scheduling tasks call this, so index ``i`` refers to the same config in
        every task. Order is stable across processes (sorted by a repr of the values).
        """
        grid = [
            p
            for p in ParameterGrid(self.param_space)
            if self.model_builder.is_valid(p, input_shape)
        ]
        grid.sort(key=lambda p: repr([p[k] for k in sorted(p)]))
        return grid

    def _part_path(self, global_dir: str):
        """Fresh per-task part-CSV path (crash-safe row stream) and its parent dir."""
        parts_dir = os.path.join(global_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        part_csv = os.path.join(parts_dir, f"{self.run_id}.csv")
        if os.path.exists(part_csv):
            os.remove(part_csv)
        return parts_dir, part_csv

    def _merge_part(self, global_dir: str, parts_dir: str, part_csv: str, rows: list) -> None:
        """Merge this task's rows into the single shared CSV and drop the part file."""
        append_rows_to_csv(global_dir, pd.DataFrame(rows))
        if os.path.exists(part_csv):
            os.remove(part_csv)
        try:
            os.rmdir(parts_dir)  # succeeds only for the last task to finish (empty dir)
        except OSError:
            pass

    def _train_one(
        self,
        params,
        model_id,
        images,
        labels,
        class_labels,
        store,
        test_images,
        test_labels,
        soft_targets=None,
        groups=None,
        is_aug=None,
    ):
        """CV-train one config, offer its artefacts to the store, return its row (or None)."""
        start = time.perf_counter()
        try:
            cv = self._cross_validate(
                params, images, labels, class_labels, soft_targets, groups=groups, is_aug=is_aug
            )
        except Exception as exc:  # defensive: a runtime failure shouldn't abort the sweep
            print(f"Model {model_id} failed to build/train ({exc}); skipping.")
            return None
        train_seconds = time.perf_counter() - start

        avg_vaf = float(np.mean(cv["val_acc_fold"]))
        avg_taf = float(np.mean(cv["train_acc_fold"]))
        row = dict(params)
        row.update(
            {
                "model_id": model_id,
                "avg_vaf": avg_vaf,
                "avg_taf": avg_taf,
                "avg_divergence": avg_taf - avg_vaf,
                "avg_val_nearmiss": float(np.mean(cv["val_nearmiss_fold"])),
                "avg_val_qwk": float(np.mean(cv["val_qwk_fold"])),
                # Real-only and augmented-only accuracy alongside the pooled figure.
                # avg_vaf stays the all-data number so it means the same thing as in every
                # other search in this project and the runs remain comparable.
                **(
                    {
                        "avg_vaf_orig": float(np.nanmean(cv["val_acc_orig_fold"])),
                        "avg_vaf_aug": float(np.nanmean(cv["val_acc_aug_fold"])),
                        "avg_taf_orig": float(np.nanmean(cv["train_acc_orig_fold"])),
                        "avg_taf_aug": float(np.nanmean(cv["train_acc_aug_fold"])),
                        "aug_minus_orig_vaf": float(
                            np.nanmean(cv["val_acc_aug_fold"]) - np.nanmean(cv["val_acc_orig_fold"])
                        ),
                    }
                    if cv["val_acc_orig_fold"]
                    else {}
                ),
                "best_epoch": cv["best_epoch"],
                "train_seconds": round(train_seconds, 1),
                "cv_seed": self.cv_seed,
            }
        )
        if test_images is not None and test_labels is not None and cv["best_model"] is not None:
            row.update(self._test_metrics(cv["best_model"], test_images, test_labels, class_labels))

        def _save(folder, cv=cv):
            self._save_artifacts(folder, cv, class_labels)

        store.offer(model_id, avg_vaf, _save)
        return row

    def run(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        test_images: np.ndarray = None,
        test_labels: np.ndarray = None,
        n_models: int = 4,
        global_dir: str = ".",
        top_n: int = 100,
        groups: np.ndarray | None = None,
        is_aug: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Random search: sample ``n_models`` configs, contributing to a shared global top-N.

        Returns this task's rows as a DataFrame (sorted by ``avg_vaf`` descending).
        """
        train_labels = np.asarray(train_labels).astype(int)
        class_labels = sorted(np.unique(train_labels).tolist())

        sampled = self._sample_valid_configs(n_models, train_images.shape[1:])
        parts_dir, part_csv = self._part_path(global_dir)
        store = TopNStore(global_dir, top_n)
        print(f"Run {self.run_id}: {len(sampled)} configs -> global top-{top_n} at {global_dir}")

        rows, header_written = [], False
        for index, params in enumerate(sampled):
            model_id = f"{self.run_id}-{index}"
            print(f"\n=== Model {model_id} ({index + 1}/{len(sampled)}) ===\n{params}")
            row = self._train_one(
                params,
                model_id,
                train_images,
                train_labels,
                class_labels,
                store,
                test_images,
                test_labels,
                groups=groups,
                is_aug=is_aug,
            )
            if row is None:
                continue
            rows.append(row)
            pd.DataFrame([row]).to_csv(part_csv, mode="a", header=not header_written, index=False)
            header_written = True

        if not rows:
            raise RuntimeError("No configs completed successfully.")
        self._merge_part(global_dir, parts_dir, part_csv, rows)
        print(
            f"Run {self.run_id}: merged {len(rows)} models into {global_dir}/random_search_results.csv"
        )
        return pd.DataFrame(rows).sort_values("avg_vaf", ascending=False).reset_index(drop=True)

    def run_grid(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        test_images: np.ndarray = None,
        test_labels: np.ndarray = None,
        global_dir: str = ".",
        top_n: int = 100,
        max_configs: int | None = None,
        max_rss_mb: int | None = None,
        soft_targets: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        is_aug: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Full grid search by self-scheduling.

        This task pulls the next unclaimed config from a shared counter (``next.txt``)
        until the whole grid is done, contributing to the same shared global top-N.
        Launch many of these as a SLURM array: together they cover every valid config
        exactly once, load-balanced automatically, and a re-submit resumes from the
        counter. Returns this task's rows.

        ``max_configs`` stops this task after that many configs and returns normally,
        leaving the rest of the grid for other tasks. Use it to bound a task's lifetime
        when memory grows with the number of models trained in one process.

        TensorFlow retains memory across repeated ``model.fit`` calls, and neither
        ``del model``, ``clear_session()`` nor ``gc.collect()`` releases it -- see
        tensorflow/tensorflow#113529, reported against TF 2.21.0, the version this
        project's container pins. Measured here at roughly 2.5 GB of host RSS per config.
        The task must therefore END before it is killed, not be killed: ``claim()``
        advances the shared counter BEFORE training, so a task that dies mid-config
        leaves that index claimed but never trained, and it is silently absent from the
        results with no error anywhere. Exiting cleanly instead runs ``_merge_part`` and
        loses nothing, and a queued array task resumes from the counter.

        ``max_rss_mb`` is the backstop for ``max_configs`` being mis-sized: the task also
        stops once its own resident set reaches that ceiling. Set it below the job's
        ``--mem`` so the task ends on its own terms rather than being killed.
        """
        train_labels = np.asarray(train_labels).astype(int)
        class_labels = sorted(np.unique(train_labels).tolist())

        configs = self.all_valid_configs(train_images.shape[1:])
        counter = JobCounter(global_dir, len(configs))
        parts_dir, part_csv = self._part_path(global_dir)
        store = TopNStore(global_dir, top_n)
        print(
            f"Grid: {len(configs)} valid configs; task {self.run_id} self-scheduling -> top-{top_n}"
        )

        rows, header_written = [], False
        while True:
            i = counter.claim()
            if i is None:
                break
            params = configs[i]
            print(f"\n=== Config {i}/{len(configs)} (task {self.run_id}) ===\n{params}")
            row = self._train_one(
                params,
                str(i),
                train_images,
                train_labels,
                class_labels,
                store,
                test_images,
                test_labels,
                soft_targets=soft_targets,
                groups=groups,
                is_aug=is_aug,
            )
            if row is None:
                continue
            row["task"] = self.run_id
            rows.append(row)
            pd.DataFrame([row]).to_csv(part_csv, mode="a", header=not header_written, index=False)
            header_written = True

            # Log RSS every config: it rises steadily (TF retains memory across fit calls,
            # tensorflow/tensorflow#113529) and this trend is what sizes max_configs.
            rss_now = _rss_mb()
            if rss_now is not None:
                print(f"  [rss] after {len(rows)} config(s): {rss_now:.0f} MB", flush=True)

            if max_configs is not None and len(rows) >= max_configs:
                print(
                    f"Task {self.run_id}: hit the {max_configs}-config budget; exiting "
                    f"cleanly so a queued task can resume from the counter."
                )
                break

            if max_rss_mb is not None and rss_now is not None and rss_now >= max_rss_mb:
                # Backstop for the config budget being mis-sized. Being OOM-killed here
                # would silently drop the next claimed config, so stop while we can still
                # merge. Costs one task restart; the alternative costs a missing result.
                print(
                    f"Task {self.run_id}: RSS {rss_now:.0f} MB reached the {max_rss_mb} MB "
                    f"ceiling after {len(rows)} configs; exiting cleanly."
                )
                break

        if rows:
            self._merge_part(global_dir, parts_dir, part_csv, rows)
        elif os.path.exists(part_csv):
            os.remove(part_csv)  # this worker started after the grid was already exhausted
        print(f"Task {self.run_id}: trained {len(rows)} configs")
        return pd.DataFrame(rows)
