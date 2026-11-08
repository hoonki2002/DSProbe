"""Training and checkpoint selection shared by the released experiments."""

from __future__ import annotations

import random
import time
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from .cap import ReferenceStepCap


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    eval_every: int = 25
    evaluate_final_epoch: bool = True
    shuffle: str = "repo"
    later_ties: bool = False
    task: str = "membership"
    cap: float = 1e-3
    num_workers: int = 2

    def __post_init__(self):
        integers = (self.epochs, self.batch_size, self.eval_every, self.num_workers)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ValueError("Epochs, batch size, evaluation interval and worker count must be integers")
        if min(self.epochs, self.batch_size, self.eval_every) <= 0:
            raise ValueError("Epochs, batch size and evaluation interval must be positive")
        if not all(math.isfinite(value) for value in (self.learning_rate, self.weight_decay, self.cap)):
            raise ValueError("Optimizer and cap settings must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.cap < 0 or self.num_workers < 0:
            raise ValueError("Invalid optimizer or cap setting")
        if self.shuffle not in {"repo", "legacy", "sd_loader"}:
            raise ValueError("Unknown shuffle policy")
        if self.task not in {"membership", "sd_lora"}:
            raise ValueError("Unknown prediction task")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BatchOrder:
    """Match the two historical Model-J sampling rules."""

    def __init__(self, count, batch_size, seed, policy):
        self.count, self.batch_size, self.seed, self.policy = count, batch_size, seed, policy
        self.generator = torch.Generator().manual_seed(seed)

    def batches(self, epoch):
        if self.policy == "legacy":
            generator = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
            indices = torch.randperm(self.count, generator=generator)
        else:
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            indices = torch.tensor(list(RandomSampler(range(self.count), generator=self.generator)))
        return indices.split(self.batch_size)


def load_bank(dataset, batch_size=128):
    """Read a layer into CPU RAM once; no weight cache is written to disk."""
    count = len(dataset)
    if not count:
        raise ValueError("Empty model bank")
    if batch_size < 1:
        raise ValueError("Bank loading batch size must be positive")
    first_weights, first_labels = dataset[0]
    first_labels = torch.as_tensor(first_labels)
    # Preallocate once: concatenating a list of full-bank chunks would keep
    # both the chunks and the complete destination alive at the same time.
    weights = torch.empty((count, *first_weights.shape), dtype=first_weights.dtype, device="cpu")
    labels = torch.empty((count, *first_labels.shape), dtype=first_labels.dtype, device="cpu")
    weights[0].copy_(first_weights)
    labels[0].copy_(first_labels)
    del first_weights, first_labels
    for start in range(1, count, batch_size):
        for index in range(start, min(start + batch_size, count)):
            sample_weights, sample_labels = dataset[index]
            sample_labels = torch.as_tensor(sample_labels)
            if sample_weights.shape != weights.shape[1:] or sample_labels.shape != labels.shape[1:]:
                raise ValueError("Inconsistent shape while loading the model bank")
            weights[index].copy_(sample_weights)
            labels[index].copy_(sample_labels)
    return weights, labels


def training_mean(bank, device="cpu"):
    weights = bank[0]
    accumulator = torch.zeros(weights.shape[1:], dtype=torch.float64, device=device)
    for start in range(0, len(weights), 128):
        accumulator += weights[start:start + 128].to(device).sum(0, dtype=torch.float64)
    return (accumulator / len(weights)).float().cpu()


def make_sd_loader(dataset, config, seed, shuffle):
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=shuffle,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=config.num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def _iterator(data, batch_size):
    if isinstance(data, tuple):
        for start in range(0, len(data[0]), batch_size):
            yield data[0][start:start + batch_size], data[1][start:start + batch_size]
    else:
        yield from data


def _loss(logits, labels, task):
    if task == "membership":
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)
    return nn.functional.cross_entropy(logits, labels)


def move_batch(batch, device):
    if len(batch) == 3:
        b, a, labels = batch
        return b.to(device) @ a.to(device), labels.to(device)
    weights, labels = batch
    return weights.to(device), labels.to(device)


def _correct(logits, labels, task):
    if task == "membership":
        return int(((logits.sigmoid() > 0.5) == labels).sum()), labels.numel()
    return int((logits.argmax(1) == labels).sum()), labels.numel()


@torch.inference_mode()
def evaluate(model, data, config, device, text=None):
    model.eval()
    loss_sum, correct, count, samples = 0.0, 0, 0, 0
    for batch in _iterator(data, config.batch_size):
        weights, labels = move_batch(batch, device)
        logits = model(weights) if text is None else model(weights, text)
        loss = _loss(logits, labels, config.task)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        hits, total = _correct(logits, labels, config.task)
        correct += hits
        count += total
        samples += len(labels)
        loss_sum += float(loss) * len(labels)
    if not count:
        raise ValueError("Empty evaluation split")
    return {"loss": loss_sum / samples, "accuracy": 100 * correct / count}


def fit(model, train, validation, tests, config, seed, device, texts=None):
    """Select one state per validation group; evaluate test only after training.

    Membership: keys are 'id'. SD: keys are 'id' and 'zero_shot'.
    The SD states are selected independently from the same training trajectory.
    Best states remain in CPU RAM; the caller may explicitly choose to save them.
    """
    device = torch.device(device)
    if set(validation) != set(tests):
        raise ValueError("Validation and test groups must match")
    texts = {} if texts is None else {key: value.to(device) for key, value in texts.items()}
    model.to(device)
    reference = getattr(model, "x_star", None)
    cap = None
    if isinstance(reference, nn.Parameter) and config.cap > 0:
        # Preserve the historical capped-reference optimizer grouping, including
        # CUDA foreach dispatch: the reference occupies its own second group.
        other_parameters = [parameter for parameter in model.parameters() if parameter is not reference]
        if not other_parameters:
            raise ValueError("A capped model needs non-reference trainable parameters")
        optimizer = torch.optim.Adam([
            {"params": other_parameters, "lr": config.learning_rate, "weight_decay": config.weight_decay},
            {"params": [reference], "lr": config.learning_rate, "weight_decay": config.weight_decay},
        ])
        cap = ReferenceStepCap(reference, reference.detach().clone(), config.cap)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    frozen_reference = reference.detach().clone() if reference is not None and not reference.requires_grad else None
    if isinstance(train, tuple):
        order = BatchOrder(len(train[0]), config.batch_size, seed, config.shuffle)
    best, states, history = {}, {}, []
    clipped_steps = 0
    maximum_applied_relative_step = 0.0
    start = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        if isinstance(train, tuple):
            batches = ((train[0][idx], train[1][idx]) for idx in order.batches(epoch))
        else:
            batches = iter(train)
        for batch in batches:
            weights, labels = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(weights) if "id" not in texts else model(weights, texts["id"])
            loss = _loss(logits, labels, config.task)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            before = cap.capture() if cap else None
            optimizer.step()
            if cap:
                stats = cap.apply(before)
                clipped_steps += int(stats["clipped"])
                maximum_applied_relative_step = max(maximum_applied_relative_step, stats["applied_relative_update"])
        if epoch % config.eval_every != 0 and not (config.evaluate_final_epoch and epoch == config.epochs - 1):
            continue
        record = {"epoch": epoch + 1}
        for group, data in validation.items():
            metrics = evaluate(model, data, config, device, texts.get(group))
            record[group] = metrics
            improved = group not in best or metrics["accuracy"] > best[group]["accuracy"]
            if config.later_ties and group in best:
                improved = metrics["accuracy"] >= best[group]["accuracy"]
            if improved:
                best[group] = {"epoch": epoch + 1, **metrics}
                states[group] = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        history.append(record)
        print(f"epoch {epoch + 1}/{config.epochs}: " + ", ".join(f"{g}={record[g]['accuracy']:.3f}" for g in validation), flush=True)
    if frozen_reference is not None and not torch.equal(reference, frozen_reference):
        raise RuntimeError("Frozen reference changed during training")
    results = {}
    for group, data in tests.items():
        model.load_state_dict(states[group], strict=True)
        metrics = evaluate(model, data, config, device, texts.get(group))
        results[group] = {"validation": best[group], "test": metrics}
    return {
        "seed": seed, "training": asdict(config), "results": results,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "history": history, "elapsed_seconds": time.perf_counter() - start,
        "reference_clipped_steps": clipped_steps,
        "maximum_applied_relative_step": maximum_applied_relative_step,
    }, states
