"""Cross-tree experiments with per-tree probes and one shared encoder/head.

Run ``python -m dsprobe.shared --help``. Defaults reproduce the supplied
cross-tree experiment's uncapped, training-mean-initialized reference.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import nn

import hashlib

from .common import (accuracy, code_hash, existing_result, fit_independent,
                       load_bank, parse_args, prepare, protocol, reference_caps,
                       result_path, save_result, update)
from .modelj_data import load_reference, resolve_layer_name
from .runner import _reference_path


# ``*_dsprobe`` subtract the frozen pretrained root W_pre; ``specialist``/``shared_ref``
# subtract a learned W* initialized at the training mean (R-DSProbe).
VARIANTS = ("specialist_raw", "specialist", "shared_raw", "shared_ref",
            "specialist_dsprobe", "shared_dsprobe")


class FrontEnd(nn.Module):
    """Four branch-specific probes/projections and an optional learned origin."""
    def __init__(self, shape, n_probes=128, reference=None, proj_dim=128, trainable_reference=True):
        super().__init__()
        d_h, d_w = shape
        self.probes_xu = nn.Linear(n_probes, d_w, bias=False)
        self.probes_xxtu = nn.Linear(n_probes, d_h, bias=False)
        self.probes_xtu = nn.Linear(n_probes, d_h, bias=False)
        self.probes_xtxu = nn.Linear(n_probes, d_w, bias=False)
        self.proj_row = nn.Linear(d_h, proj_dim, bias=False)
        self.proj_row2 = nn.Linear(d_h, proj_dim, bias=False)
        self.proj_col = nn.Linear(d_w, proj_dim, bias=False)
        self.proj_col2 = nn.Linear(d_w, proj_dim, bias=False)
        if reference is None:
            self.W_star = None
        elif trainable_reference:
            self.W_star = nn.Parameter(reference.clone())
        else:                                           # DSProbe: a fixed, untrained root
            self.register_buffer("W_star", reference.clone())

    def responses(self, x):
        xt = x.transpose(1, 2)
        xu = x @ self.probes_xu.weight
        xxtu = x @ (xt @ self.probes_xxtu.weight)
        xtu = xt @ self.probes_xtu.weight
        xtxu = xt @ (x @ self.probes_xtxu.weight)
        if self.W_star is not None:
            w, wt = self.W_star, self.W_star.transpose(0, 1)
            xu = xu - w @ self.probes_xu.weight
            xtu = xtu - wt @ self.probes_xtu.weight
            xxtu = xxtu - w @ (wt @ self.probes_xxtu.weight)
            xtxu = xtxu - wt @ (w @ self.probes_xtxu.weight)
        return xu, xxtu, xtu, xtxu

    def encode_responses(self, responses):
        encoded = []
        for response, projection in zip(responses, (self.proj_row, self.proj_row2,
                                                    self.proj_col, self.proj_col2)):
            if response.shape[1] * response.shape[2] < 2:
                raise ValueError("Response standardization requires at least two entries")
            standardized = ((response - response.mean(dim=(1, 2), keepdim=True)) /
                            (response.std(dim=(1, 2), keepdim=True) + 1e-8))
            encoded.append(torch.relu(projection(standardized.transpose(1, 2))).reshape(len(response), -1))
        return torch.cat(encoded, dim=1)

    def forward(self, x):
        return self.encode_responses(self.responses(x))


class SpecialistProbe(nn.Module):
    def __init__(self, shape, n_probes=128, reference=None, proj_dim=128, rep_dim=512,
                 trainable_reference=True):
        super().__init__()
        self.front = FrontEnd(shape, n_probes, reference, proj_dim, trainable_reference)
        self.encoder = nn.Linear(4 * proj_dim * n_probes, rep_dim)
        self.head = nn.Linear(rep_dim, 100)

    def forward(self, x):
        return self.head(self.encoder(self.front(x)))


class SharedProbe(nn.Module):
    def __init__(self, shapes, n_probes=128, references=None, proj_dim=128, rep_dim=512,
                 trainable_reference=True):
        super().__init__()
        self.fronts = nn.ModuleDict({name: FrontEnd(shape, n_probes,
                                  references[name] if references is not None else None, proj_dim,
                                  trainable_reference)
                                    for name, shape in shapes.items()})
        self.encoder = nn.Linear(4 * proj_dim * n_probes, rep_dim)
        self.head = nn.Linear(rep_dim, 100)

    def forward(self, name, x):
        return self.head(self.encoder(self.fronts[name](x)))

    def parameter_report(self):
        shared = sum(p.numel() for name, p in self.named_parameters() if not name.startswith("fronts."))
        per_tree = {name: sum(p.numel() for p in front.parameters()) for name, front in self.fronts.items()}
        return {"shared_encoder_head": shared, "per_tree_frontend": per_tree,
                "total": shared + sum(per_tree.values())}


def fit_shared(model, data, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    caps = reference_caps(model, args.cap)
    steps = max(len(data[name]["train"][0]) for name in args.models) // args.per_tree_batch
    if steps < 1:
        raise ValueError("The shared schedule requires max(train sizes) >= per_tree_batch")
    best, selected_epoch, state, history = -1.0, None, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutations = {name: torch.randperm(len(data[name]["train"][0]), device=args.device)
                        for name in args.models}
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = 0
            for name in args.models:
                x, y = data[name]["train"]
                indices = permutations[name][step * args.per_tree_batch:(step + 1) * args.per_tree_batch]
                if len(indices):
                    loss = loss + criterion(model(name, x[indices]), y[indices])
            update(optimizer, loss, caps)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            values = {name: accuracy(lambda x, n=name: model(n, x), *data[name]["val"],
                                     args.eval_batch_size) for name in args.models}
            value = float(np.mean(list(values.values())))
            history.append({"epoch": epoch, "validation_accuracy": values, "macro_accuracy": value})
            if value > best:
                best, selected_epoch = value, epoch
                state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    test = {name: accuracy(lambda x, n=name: model(n, x), *data[name]["test"], args.eval_batch_size)
            for name in args.models}
    return {"test_accuracy": test, "macro_accuracy": float(np.mean(list(test.values()))),
            "validation_macro_accuracy": best, "selected_epoch": selected_epoch,
            "parameter_report": model.parameter_report(), "history": history}


def main(argv=None):
    args = parse_args("shared", VARIANTS, argv)
    if not prepare(args):
        return
    data, shapes, fingerprints = load_bank(args.data_root, args.models, args.layers, args.device)
    refs = {name: data[name]["train"][0].mean(0) for name in args.models}
    roots = {}
    if any(v.endswith("_dsprobe") for v in args.variants):
        for name in args.models:
            key = resolve_layer_name(name, args.layers[name])
            roots[name] = load_reference(_reference_path(args.reference_root, name, args.layers[name]),
                                         name, key, shapes[name])
    source = code_hash()
    for variant in args.variants:
        for seed in args.seeds:
            contract = {"protocol": protocol(args, variant), "seed": seed,
                        "data": fingerprints, "code_sha256": source}
            if variant.endswith("_dsprobe"):
                contract["pretrained_roots_sha256"] = {
                    name: hashlib.sha256(root.contiguous().numpy().tobytes()).hexdigest()
                    for name, root in roots.items()}
            path = result_path(args, variant, seed)
            if existing_result(path, contract, args.resume) is not None:
                print(f"Resume: {variant} seed {seed} already complete", flush=True)
                continue
            start = time.perf_counter()
            np.random.seed(seed)
            if variant.startswith("specialist"):
                details = {}
                for name in args.models:
                    torch.manual_seed(seed)
                    reference = (None if variant == "specialist_raw" else
                                 roots[name] if variant == "specialist_dsprobe" else refs[name])
                    model = SpecialistProbe(shapes[name], args.n_probes, reference,
                                            args.proj_dim, args.rep_dim,
                                            variant != "specialist_dsprobe").to(args.device)
                    details[name] = fit_independent(model, data[name], args)
                    del model
                test = {name: value["test_accuracy"] for name, value in details.items()}
                result = {"test_accuracy": test, "macro_accuracy": float(np.mean(list(test.values()))),
                          "details": details, "parameters": sum(v["parameters"] for v in details.values())}
            else:
                torch.manual_seed(seed)
                references = (refs if variant == "shared_ref" else
                              roots if variant == "shared_dsprobe" else None)
                model = SharedProbe(shapes, args.n_probes, references, args.proj_dim, args.rep_dim,
                                    variant != "shared_dsprobe").to(args.device)
                result = fit_shared(model, data, args)
                del model
            result.update(contract=contract, complete=True, elapsed_seconds=time.perf_counter() - start)
            save_result(path, result)
            print(json.dumps({"variant": variant, "seed": seed, "test_accuracy": result["test_accuracy"],
                              "macro_accuracy": result["macro_accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
