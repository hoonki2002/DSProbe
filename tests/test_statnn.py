"""CPU-only regression tests for the all-parameter StatNN reproduction.

All checkpoint banks and result fixtures are temporary. The only fitted model
is a tiny, three-round, 100-output XGBoost classifier; real Model-J data and GPUs
are never used.
"""
from __future__ import annotations

import contextlib
import builtins
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from dsprobe import statnn
from dsprobe.analysis import aggregate, collect_results, summarize
from dsprobe.cli import main as cli_main, parser as cli_parser
from dsprobe.modelj_data import CIFAR100_CLASSES


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("ResNet", "SupViT", "MAE", "DINO")
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None


def checkpoint_row(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def make_cache_leaf(root: Path, split: str, index: int, *, model="SupViT",
                    physical_name=None, declared_split=None, targets=None) -> Path:
    """Make an HF-style snapshot; its lexical and resolved orders may differ."""
    folder = root / model / split / f"models--fixture--model_idx_{index:04d}"
    snapshot = folder / "snapshots" / "v1" / "model.safetensors"
    snapshot.parent.mkdir(parents=True)
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text("v1", encoding="utf-8")
    if physical_name is None:
        physical = snapshot
    else:
        physical = root / "physical" / physical_name
        physical.parent.mkdir(parents=True, exist_ok=True)
    chosen = list(CIFAR100_CLASSES[::2]) if targets is None else targets
    save_file({"a.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
              str(physical), metadata={
                  "split": declared_split or split, "model_idx": str(index),
                  "dataset_chosen_targets": repr(chosen),
              })
    if physical != snapshot:
        snapshot.symlink_to(physical)
    return snapshot


def make_inventory_bank(root: Path) -> None:
    # Cache-directory order is a,z while resolved source order is z,a.
    for split, base in zip(statnn.SPLITS, (0, 10, 20)):
        make_cache_leaf(root, split, base, physical_name=f"{split}_z.safetensors")
        make_cache_leaf(root, split, base + 1, physical_name=f"{split}_a.safetensors")


class StatisticTests(unittest.TestCase):
    def test_original_statistics_order_and_population_standard_deviation(self):
        values = np.asarray([0, 1, 2, 3], dtype=np.float32)
        got = statnn.tensor_statistics(values)
        self.assertEqual(got.shape, (7,))
        self.assertEqual(got.dtype, np.float64)
        expected = np.asarray([0, .75, 1.5, 2.25, 3, np.std(values, ddof=0), np.mean(values)])
        np.testing.assert_array_equal(got, expected)
        self.assertNotEqual(got[5], float(np.var(values)))
        self.assertNotEqual(got[5], float(np.std(values, ddof=1)))
        np.testing.assert_array_equal(statnn.tensor_statistics(values[::-1]), got)

    def test_float32_reduction_semantics_are_preserved(self):
        values = np.asarray([1e8, 1, -1e8], dtype=np.float32)
        got = statnn.tensor_statistics(values)
        self.assertEqual(got[5], float(np.std(values, ddof=0)))
        self.assertEqual(got[6], float(np.mean(values)))
        self.assertNotEqual(got[6], float(np.mean(values.astype(np.float64))))

    def test_parameter_schema_covers_named_parameters_and_excludes_buffers(self):
        for model in MODELS:
            with self.subTest(model=model):
                schema = statnn.parameter_schema(model)
                keys = schema["parameter_keys"]
                self.assertEqual(len(keys), 314 if model == "ResNet" else 200)
                self.assertEqual(keys, sorted(set(keys)))
                self.assertEqual(set(keys), set(schema["shapes"]))
                self.assertTrue(any(key.endswith("bias") for key in keys))
                self.assertTrue(any("classifier" in key for key in keys))
                self.assertFalse(any(token in key for key in keys for token in
                                     ("running_mean", "running_var", "num_batches_tracked")))
                self.assertTrue(all(all(int(size) > 0 for size in schema["shapes"][key])
                                    for key in keys))
        with self.assertRaises((ValueError, KeyError)):
            statnn.parameter_schema("NotAModel")

    def test_per_parameter_concatenation_ignores_buffers_and_metadata(self):
        schema = {"parameter_keys": ["a.normalization.weight", "z.bias"],
                  "shapes": {"a.normalization.weight": [2, 2], "z.bias": [2]}}
        weights = {"z.bias": torch.tensor([10., 30.]),
                   "a.normalization.weight": torch.tensor([[0., 1.], [2., 3.]]),
                   "a.normalization.running_mean": torch.tensor([float("nan")]),
                   "a.normalization.num_batches_tracked": torch.tensor(77)}
        with tempfile.TemporaryDirectory(prefix="statnn-features-") as directory:
            path = Path(directory) / "model.safetensors"
            save_file(weights, str(path), metadata={"irrelevant_answer": "do not use"})
            got = statnn.extract_checkpoint(checkpoint_row(path), schema)
        expected = np.concatenate([statnn.tensor_statistics(weights[key].numpy().reshape(-1))
                                   for key in schema["parameter_keys"]]).astype(np.float32)
        self.assertEqual(got.dtype, np.float32)
        self.assertEqual(got.shape, (14,))
        np.testing.assert_array_equal(got, expected)

    def test_checkpoint_missing_key_wrong_shape_and_nonfinite_parameters_rejected(self):
        schema = {"parameter_keys": ["a.weight", "z.bias"],
                  "shapes": {"a.weight": [2, 2], "z.bias": [2]}}
        cases = (
            {"a.weight": torch.ones(2, 2)},
            {"a.weight": torch.ones(4), "z.bias": torch.ones(2)},
            {"a.weight": torch.full((2, 2), float("nan")), "z.bias": torch.ones(2)},
            {"a.weight": torch.ones(2, 2), "z.bias": torch.ones(2), "extra.weight": torch.ones(1)},
        )
        with tempfile.TemporaryDirectory(prefix="statnn-invalid-") as directory:
            for index, weights in enumerate(cases):
                with self.subTest(case=index):
                    path = Path(directory) / f"{index}.safetensors"
                    save_file(weights, str(path))
                    with self.assertRaises((ValueError, KeyError)):
                        statnn.extract_checkpoint(checkpoint_row(path), schema)


class InventoryTests(unittest.TestCase):
    def test_inventory_sorts_resolved_paths_and_builds_100_bit_targets(self):
        with tempfile.TemporaryDirectory(prefix="statnn-inventory-") as directory:
            root = Path(directory)
            make_inventory_bank(root)
            found = statnn.inventory(root, "SupViT")
            self.assertEqual(tuple(found), statnn.SPLITS)
            for split, rows in found.items():
                self.assertEqual(len(rows), 2)
                self.assertEqual([r["path"] for r in rows], sorted(r["path"] for r in rows))
                self.assertTrue(rows[0]["path"].endswith(f"{split}_a.safetensors"))
                for row in rows:
                    self.assertEqual(Path(row["path"]), Path(row["path"]).resolve())
                    self.assertEqual(row["labels"], [int(i % 2 == 0) for i in range(100)])
                    self.assertEqual(len(row["labels"]), 100)
                    self.assertEqual(sum(row["labels"]), 50)
                    self.assertIn("model_idx", row)
                    self.assertEqual(row["size_bytes"], Path(row["path"]).stat().st_size)
                    self.assertEqual(row["mtime_ns"], Path(row["path"]).stat().st_mtime_ns)

    def test_inventory_rejects_source_reused_across_splits(self):
        with tempfile.TemporaryDirectory(prefix="statnn-leak-") as directory:
            root = Path(directory)
            train = make_cache_leaf(root, "train", 0)
            val = make_cache_leaf(root, "val", 1)
            make_cache_leaf(root, "test", 2)
            val.unlink()  # Only generated temporary fixtures are replaced.
            val.symlink_to(train)
            with self.assertRaises(ValueError):
                statnn.inventory(root, "SupViT")

    def test_inventory_rejects_wrong_split_metadata_and_bad_membership_labels(self):
        for bad_kind in ("split", "too_few", "duplicates", "unknown"):
            with self.subTest(case=bad_kind), tempfile.TemporaryDirectory(prefix="statnn-labels-") as directory:
                root = Path(directory)
                kwargs = {}
                if bad_kind == "split":
                    kwargs["declared_split"] = "test"
                elif bad_kind == "too_few":
                    kwargs["targets"] = list(CIFAR100_CLASSES[:49])
                elif bad_kind == "duplicates":
                    kwargs["targets"] = [CIFAR100_CLASSES[0]] * 50
                else:
                    kwargs["targets"] = list(CIFAR100_CLASSES[:49]) + ["unknown"]
                make_cache_leaf(root, "train", 0, **kwargs)
                make_cache_leaf(root, "val", 1)
                make_cache_leaf(root, "test", 2)
                with self.assertRaises(ValueError):
                    statnn.inventory(root, "SupViT")


class ClassifierTests(unittest.TestCase):
    def test_missing_sklearn_reports_optional_statnn_install_requirement(self):
        original_import = builtins.__import__

        def import_without_sklearn(name, *args, **kwargs):
            if name == "sklearn":
                raise ImportError("No module named 'sklearn'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_sklearn):
            with self.assertRaisesRegex(RuntimeError, r"\.\[statnn\]"):
                statnn._xgboost()

    def test_multilabel_metrics_use_threshold_inclusive_half_and_mean_binary_logloss(self):
        labels = np.tile(np.arange(100) % 2, (2, 1)).astype(np.uint8)
        probabilities = np.where(labels, .8, .2).astype(np.float32)
        probabilities[0, 0] = .5  # Inclusive >= .5 makes this one bit wrong.
        metrics = statnn.multilabel_metrics(probabilities, labels)
        self.assertAlmostEqual(metrics["bit_accuracy"], 199 / 200)
        self.assertAlmostEqual(metrics["exact_match_accuracy"], .5)
        p = np.clip(probabilities.astype(np.float64), 1e-12, 1 - 1e-12)
        expected = -np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))
        self.assertAlmostEqual(metrics["binary_log_loss"], float(expected))
        with self.assertRaises(ValueError):
            statnn.multilabel_metrics(probabilities[:1], labels)

    def test_original_boosting_hyperparameters(self):
        expected = {"n_estimators": 500, "early_stopping_rounds": 25, "learning_rate": .05,
                    "max_depth": 3, "subsample": .9, "colsample_bytree": 1.,
                    "reg_lambda": 1., "reg_alpha": 0., "tree_method": "hist", "device": "cpu",
                    "n_jobs": 4}
        self.assertEqual(set(statnn.DEFAULT_HYPERPARAMETERS), set(expected))
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(statnn.DEFAULT_HYPERPARAMETERS[key], value)

    @unittest.skipUnless(HAS_XGBOOST, "Optional xgboost dependency is not installed")
    def test_real_cpu_xgboost_fit_has_100_outputs_and_only_validation_evalset(self):
        import xgboost
        self.assertEqual(xgboost.__version__, "3.2.0")
        rng = np.random.default_rng(17)
        features = {split: rng.normal(size=(count, 14)).astype(np.float32)
                    for split, count in zip(statnn.SPLITS, (12, 6, 8))}
        labels = {split: ((np.arange(len(x))[:, None] + np.arange(100)[None, :]) % 2).astype(np.uint8)
                  for split, x in features.items()}
        hyperparameters = dict(statnn.DEFAULT_HYPERPARAMETERS,
                               n_estimators=3, early_stopping_rounds=2, n_jobs=1)
        original_fit = xgboost.XGBClassifier.fit
        calls = []

        def verified_fit(classifier, x, y, **kwargs):
            self.assertIs(x, features["train"])
            self.assertIs(y, labels["train"])
            self.assertEqual(len(kwargs["eval_set"]), 1)
            self.assertIs(kwargs["eval_set"][0][0], features["val"])
            self.assertIs(kwargs["eval_set"][0][1], labels["val"])
            calls.append(kwargs)
            return original_fit(classifier, x, y, **kwargs)

        with patch.object(xgboost.XGBClassifier, "fit", new=verified_fit):
            classifier, val, test = statnn.train_classifier(features, labels, 3, hyperparameters)
        self.assertEqual(len(calls), 1)
        self.assertEqual(val.shape, (6, 100))
        self.assertEqual(test.shape, (8, 100))
        self.assertTrue(np.isfinite(val).all() and np.isfinite(test).all())
        self.assertTrue(((test >= 0) & (test <= 1)).all())
        self.assertLess(classifier.best_iteration, 3)
        params = classifier.get_params()
        self.assertEqual(params["device"], "cpu")
        self.assertEqual(params["objective"], "binary:logistic")
        self.assertEqual(params["eval_metric"], "logloss")
        self.assertEqual(params["multi_strategy"], "one_output_per_tree")
        np.testing.assert_array_equal(val, classifier.predict_proba(features["val"]))
        np.testing.assert_array_equal(test, classifier.predict_proba(features["test"]))


class CLITests(unittest.TestCase):
    def test_default_statnn_suite_dry_run_has_20_full_model_jobs_without_loading_bank(self):
        stream = io.StringIO()
        with patch.object(statnn, "inventory", side_effect=AssertionError("dry-run read dataset")), \
                contextlib.redirect_stdout(stream):
            cli_main(["train", "--suite", str(ROOT / "configs/statnn.json"), "--dry-run"])
        text = stream.getvalue()
        self.assertIn("20 runs", text)
        self.assertEqual(sum(line.startswith(tuple(model + "/" for model in MODELS))
                             for line in text.splitlines()), 20)
        self.assertIn("full_model", text)
        for model in MODELS:
            self.assertIn(model, text)
        self.assertNotIn("layer_", text)

    def test_model_seed_filters_remain_available(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            cli_main(["train", "--suite", str(ROOT / "configs/statnn.json"),
                      "--models", "MAE", "--seeds", "2,4", "--threads", "1",
                      "--num-workers", "0", "--device", "cpu", "--resume", "--dry-run"])
        text = stream.getvalue()
        self.assertIn("MAE", text)
        self.assertNotIn("ResNet", text)
        self.assertIn("2 runs", text)
        self.assertIn("seed_2", text)
        self.assertIn("seed_4", text)

    def test_legacy_neural_flags_and_cuda_are_rejected_even_on_dry_run(self):
        invalid = (["--epochs", "1"], ["--layers", "59"], ["--lr", ".001"],
                   ["--reference-root", "unused"], ["--n-probes", "4"],
                   ["--rep-dim", "8"], ["--cap", "0"], ["--cap", "0.001"],
                   ["--learned-init", "mean"], ["--device", "cuda:0"])
        for flags in invalid:
            with self.subTest(flags=flags), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises((ValueError, SystemExit)):
                    cli_main(["train", "--suite", str(ROOT / "configs/statnn.json"),
                              "--dry-run", *flags])

    @unittest.skipUnless(HAS_XGBOOST, "Optional xgboost dependency is not installed")
    def test_tiny_suite_writes_full_model_result_and_resumes_without_refitting(self):
        schema = {"parameter_keys": ["a.weight"], "shapes": {"a.weight": [2, 2]}}
        suite = {"task": "membership", "input_scope": "all_named_parameters",
                 "models": ["SupViT"], "methods": ["statnn"], "seeds": [1],
                 "hyperparameters": dict(statnn.DEFAULT_HYPERPARAMETERS,
                                         n_estimators=3, early_stopping_rounds=2, n_jobs=1)}
        with tempfile.TemporaryDirectory(prefix="statnn-suite-") as directory:
            root = Path(directory)
            for split, base in zip(statnn.SPLITS, (0, 10, 20)):
                for index in range(2):
                    make_cache_leaf(root / "data", split, base + index,
                                    targets=list(CIFAR100_CLASSES[index::2]))
            args = cli_parser().parse_args([
                "train", "--suite", str(ROOT / "configs/statnn.json"),
                "--data-root", str(root / "data"), "--output", str(root / "results"),
                "--device", "cpu", "--threads", "1", "--num-workers", "0",
            ])
            with patch.object(statnn, "parameter_schema", return_value=schema), \
                    contextlib.redirect_stdout(io.StringIO()):
                statnn.run_suite(args, suite)
                destination = root / "results" / "SupViT" / "full_model" / "statnn" / "seed_1"
                result = json.loads((destination / "result.json").read_text())
                self.assertEqual((result["method"], result["layer"], result["seed"]),
                                 (statnn.METHOD, "full_model", 1))
                self.assertIsNone(result["parameters"])
                self.assertEqual(result["contract"]["input_scope"], "all_named_parameters")
                self.assertEqual(result["results"]["id"]["validation"]["epoch"],
                                 result["best_iteration"])
                self.assertTrue((destination / "model.ubj").is_file())
                with np.load(destination / "predictions.npz", allow_pickle=False) as predictions:
                    self.assertEqual(predictions["test"].shape, (2, 100))
                    metrics = statnn.multilabel_metrics(predictions["test"], predictions["test_labels"])
                    self.assertAlmostEqual(result["results"]["id"]["test"]["accuracy"],
                                           100 * metrics["bit_accuracy"])
                summary = aggregate(collect_results(root / "results"))
                self.assertEqual(len(summary), 1)
                self.assertEqual(summary[0]["layer"], "full_model")
                self.assertAlmostEqual(summary[0]["test_mean"], result["results"]["id"]["test"]["accuracy"])
                args.resume = True
                with patch.object(statnn, "train_classifier", side_effect=AssertionError("resume refit")):
                    statnn.run_suite(args, suite)
                changed = deepcopy(suite)
                changed["hyperparameters"]["max_depth"] += 1
                with self.assertRaises(ValueError):
                    statnn.run_suite(args, changed)
                cache = destination.parent / "features.npz"
                with np.load(cache, allow_pickle=False) as stored:
                    damaged = {key: stored[key].copy() for key in stored.files}
                damaged["train_x"][0, 0] += 1
                np.savez_compressed(cache, **damaged)
                with self.assertRaises(ValueError):
                    statnn.run_suite(args, suite)


class AnalysisTests(unittest.TestCase):
    def test_full_model_results_aggregate_with_numeric_layers(self):
        base = {"task": "membership", "model": "SupViT", "layer": "full_model",
                "method": statnn.METHOD, "seed": 1, "parameters": None,
                "contract": {"input_scope": "all_named_parameters",
                             "model": {"family": statnn.METHOD},
                             "training": dict(statnn.DEFAULT_HYPERPARAMETERS)},
                "results": {"id": {"validation": {"accuracy": 60., "epoch": 2},
                                    "test": {"accuracy": 65.}}}}
        with tempfile.TemporaryDirectory(prefix="statnn-analysis-") as directory:
            root = Path(directory)
            for seed in (1, 2):
                item = deepcopy(base)
                item["seed"] = seed
                item["results"]["id"]["test"]["accuracy"] += seed - 1
                folder = root / "SupViT" / "full_model" / "statnn" / f"seed_{seed}"
                folder.mkdir(parents=True)
                (folder / "result.json").write_text(json.dumps(item))
            layer = deepcopy(base)
            layer.update(layer=59, method="mvprobe", parameters=17)
            layer["contract"]["model"] = {"family": "mvprobe"}
            folder = root / "SupViT" / "layer_59" / "mvprobe" / "seed_1"
            folder.mkdir(parents=True)
            (folder / "result.json").write_text(json.dumps(layer))
            rows = aggregate(collect_results(root))
            self.assertEqual(len(rows), 2)
            observed = next(row for row in rows if row["method"] == statnn.METHOD)
            self.assertEqual(observed["layer"], "full_model")
            self.assertEqual(observed["n_seeds"], 2)
            self.assertAlmostEqual(observed["test_mean"], 65.5)
            self.assertAlmostEqual(observed["test_sd"], .5)
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(root, root / "summary")
            self.assertTrue((root / "summary" / "summary.csv").is_file())
            self.assertIn("full_model", (root / "summary" / "results.md").read_text())


if __name__ == "__main__":
    unittest.main()
