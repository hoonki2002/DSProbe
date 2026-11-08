"""Independent response-offset control and matched raw/reference comparators.

Offsets are added before response standardization. They are trainable response
matrices, not a single weight-space origin. ``mean_response`` initializes each
offset to the negative response of the training-mean weight under its initial
probe, then trains it freely.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import nn

from .shared import FrontEnd
from .common import (code_hash, existing_result, fit_independent, load_bank,
                       parse_args, prepare, protocol, result_path, save_result)


VARIANTS = ("none", "zero", "mean_response", "learned_reference")


class OffsetProbe(nn.Module):
    def __init__(self, shape, n_probes=128, offset="none", reference=None,
                 proj_dim=128, rep_dim=512):
        super().__init__()
        if offset not in VARIANTS:
            raise ValueError(f"Unknown offset variant: {offset}")
        if offset in ("mean_response", "learned_reference") and reference is None:
            raise ValueError("This variant needs a training-mean reference")
        self.front = FrontEnd(shape, n_probes,
                              reference if offset == "learned_reference" else None, proj_dim)
        self.encoder = nn.Linear(4 * proj_dim * n_probes, rep_dim)
        self.head = nn.Linear(rep_dim, 100)
        self.offset = offset
        if offset in ("zero", "mean_response"):
            d_h, d_w = shape
            self.B_xu = nn.Parameter(torch.zeros(d_h, n_probes))
            self.B_xxtu = nn.Parameter(torch.zeros(d_h, n_probes))
            self.B_xtu = nn.Parameter(torch.zeros(d_w, n_probes))
            self.B_xtxu = nn.Parameter(torch.zeros(d_w, n_probes))
            if offset == "mean_response":
                self.init_offset_at(reference)

    @torch.no_grad()
    def init_offset_at(self, reference):
        w = reference.to(self.front.probes_xu.weight)
        wt = w.transpose(0, 1)
        self.B_xu.copy_(-(w @ self.front.probes_xu.weight))
        self.B_xxtu.copy_(-(w @ (wt @ self.front.probes_xxtu.weight)))
        self.B_xtu.copy_(-(wt @ self.front.probes_xtu.weight))
        self.B_xtxu.copy_(-(wt @ (w @ self.front.probes_xtxu.weight)))

    def forward(self, x):
        responses = self.front.responses(x)
        if self.offset in ("zero", "mean_response"):
            responses = tuple(response + offset for response, offset in zip(
                responses, (self.B_xu, self.B_xxtu, self.B_xtu, self.B_xtxu)))
        return self.head(self.encoder(self.front.encode_responses(responses)))


def main(argv=None):
    args = parse_args("offset", VARIANTS, argv)
    if not prepare(args):
        return
    if args.cap > 0 and "learned_reference" not in args.variants:
        raise ValueError("--cap applies only to learned_reference; it cannot clip free response offsets")
    data, shapes, fingerprints = load_bank(args.data_root, args.models, args.layers, args.device)
    refs = {name: data[name]["train"][0].mean(0) for name in args.models}
    source = code_hash()
    for variant in args.variants:
        for seed in args.seeds:
            contract = {"protocol": protocol(args, variant), "seed": seed,
                        "data": fingerprints, "code_sha256": source}
            path = result_path(args, variant, seed)
            if existing_result(path, contract, args.resume) is not None:
                print(f"Resume: {variant} seed {seed} already complete", flush=True)
                continue
            start = time.perf_counter()
            details = {}
            for name in args.models:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = OffsetProbe(shapes[name], args.n_probes, variant, refs[name],
                                    args.proj_dim, args.rep_dim).to(args.device)
                if variant == "mean_response":
                    # Match initialization on the training device after the
                    # default CPU initialization of the probe network.
                    model.init_offset_at(refs[name])
                details[name] = fit_independent(model, data[name], args)
                details[name]["offset_parameters"] = sum(p.numel() for key, p in model.named_parameters()
                                                         if key.startswith("B_"))
                del model
            test = {name: value["test_accuracy"] for name, value in details.items()}
            result = {"contract": contract, "complete": True, "test_accuracy": test,
                      "macro_accuracy": float(np.mean(list(test.values()))), "details": details,
                      "elapsed_seconds": time.perf_counter() - start}
            save_result(path, result)
            print(json.dumps({"variant": variant, "seed": seed, "test_accuracy": test}), flush=True)


if __name__ == "__main__":
    main()
