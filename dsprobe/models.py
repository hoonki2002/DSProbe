"""Weight-space probing models used in the accompanying experiments.

All matrix polynomials are evaluated associatively: no square Gram matrix is
formed. Four-view concatenation order is XU, XX^T U, X^T U, X^T XU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings; the experiment protocol chooses their values.

    ``probex`` is the original unstandardized single-view baseline and always
    uses the default Linear initialization. ``single`` exposes independently
    parameterized geometry ablations and honors ``standardize``/``probe_init``.
    The SD-LoRA protocol uses default Linear initialization for every family.
    Supplying a reference to a linear/transpose single view makes it residual.
    """

    input_shape: tuple[int, int]
    num_classes: int = 100
    n_probes: int = 128
    proj_dim: int = 128
    rep_dim: int = 512
    family: str = "mvprobe"
    single_response: str = "linear"
    reference_trainable: bool = False
    standardize: bool = True
    probe_init: str = "xavier"
    task: str = "membership"
    clip_dim: int = 768

    def __post_init__(self) -> None:
        families = {
            "mvprobe", "gramdiff", "probex", "single", "poly6", "poly8",
            "duplicate6", "duplicate8",
        }
        responses = {"linear", "transpose", "gram_left", "gram_right", "diff_left", "diff_right"}
        if self.family not in families:
            raise ValueError(f"unknown model family: {self.family!r}")
        if self.single_response not in responses:
            raise ValueError(f"unknown single response: {self.single_response!r}")
        if self.probe_init not in {"xavier", "default"}:
            raise ValueError("probe_init must be 'xavier' or 'default'")
        if self.task not in {"membership", "sd_lora"}:
            raise ValueError("task must be 'membership' or 'sd_lora'")
        dimensions = (*self.input_shape, self.num_classes, self.n_probes,
                      self.proj_dim, self.rep_dim, self.clip_dim)
        if len(self.input_shape) != 2 or any(int(v) != v or v < 1 for v in dimensions):
            raise ValueError("all architecture dimensions must be positive integers")


def _standardize(response: torch.Tensor) -> torch.Tensor:
    # Keep the sample-standard-deviation correction and epsilon placement.
    mean = response.mean(dim=(1, 2), keepdim=True)
    std = response.std(dim=(1, 2), keepdim=True)
    return (response - mean) / (std + 1e-8)


def _project(response: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
    projected = torch.relu(projection(response.transpose(1, 2)))
    return projected.reshape(projected.shape[0], -1)


class _PredictionModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_shape = tuple(config.input_shape)
        self.rep_dim_out = config.rep_dim

    def _create_head(self) -> None:
        if self.config.task == "membership":
            self.classification_head = nn.Linear(self.config.rep_dim, self.config.num_classes)
        else:
            self.regression_head = nn.Linear(self.config.clip_dim, self.config.rep_dim)

    def _initialize_probes(self, probes) -> None:
        if self.config.task != "sd_lora" and self.config.probe_init == "xavier":
            for probe in probes:
                nn.init.xavier_uniform_(probe.weight)

    def _register_reference(self, reference: torch.Tensor) -> None:
        if tuple(reference.shape) != self.input_shape:
            raise ValueError(f"reference must have shape {self.input_shape}")
        if not torch.isfinite(reference).all():
            raise ValueError("reference contains non-finite values")
        reference = reference.detach().to(dtype=torch.float32).contiguous()
        self.register_buffer("initial_reference", reference.clone())
        if self.config.reference_trainable:
            self.x_star = nn.Parameter(reference.clone())
        else:
            self.register_buffer("x_star", reference.clone())

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"expected weight batches shaped (B, {self.input_shape})")
        if not torch.is_floating_point(x):
            raise TypeError("weights must be floating point")

    def forward(self, x: torch.Tensor, clip_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        representation = self.extract_representation(x)
        if self.config.task == "membership":
            if clip_embeddings is not None:
                raise ValueError("clip_embeddings are only used for sd_lora")
            return self.classification_head(representation)
        if clip_embeddings is None:
            raise ValueError("sd_lora forward requires clip_embeddings")
        clip_z = self.regression_head(clip_embeddings)
        # Match the original cosine head, including its normalization rule.
        clip_z = clip_z / clip_z.norm(dim=-1, keepdim=True)
        representation = representation / representation.norm(dim=-1, keepdim=True)
        return representation @ clip_z.T


class ProbeX(_PredictionModel):
    """Single XU branch; K=512 is the fourfold-probe capacity baseline."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        d_l, d_r = self.input_shape
        self.probes = nn.Linear(config.n_probes, d_r, bias=False)
        self.shared_probe_proj = nn.Linear(d_l, config.proj_dim, bias=False)
        self.per_probe_encoder = nn.Linear(config.proj_dim * config.n_probes, config.rep_dim)
        self._create_head()

    def extract_representation(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return self.per_probe_encoder(_project(x @ self.probes.weight, self.shared_probe_proj))


class SingleBranch(_PredictionModel):
    """One probe, one projection and one encoder, without inactive branches."""

    def __init__(self, config: ModelConfig, reference: torch.Tensor | None) -> None:
        super().__init__(config)
        response = config.single_response
        d_l, d_r = self.input_shape
        left = response in {"linear", "gram_left", "diff_left"}
        first_order = response in {"linear", "transpose"}
        response_dim = d_l if left else d_r
        probe_dim = (d_r if left else d_l) if first_order else response_dim
        if response.startswith("diff_") and reference is None:
            raise ValueError("Gram-difference responses require a reference")
        if reference is not None:
            if response in {"gram_left", "gram_right"}:
                raise ValueError("raw Gram responses do not use a reference")
            self._register_reference(reference)
        self.probe = nn.Linear(config.n_probes, probe_dim, bias=False)
        self._initialize_probes((self.probe,))
        self.projection = nn.Linear(response_dim, config.proj_dim, bias=False)
        self.encoder = nn.Linear(config.n_probes * config.proj_dim, config.rep_dim)
        self._create_head()

    def raw_probe_response(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        kind, u = self.config.single_response, self.probe.weight
        if kind in {"linear", "transpose"}:
            delta = x - self.x_star.unsqueeze(0) if hasattr(self, "x_star") else x
            return delta @ u if kind == "linear" else delta.transpose(1, 2) @ u
        if kind == "gram_left":
            return x @ (x.transpose(1, 2) @ u)
        if kind == "gram_right":
            return x.transpose(1, 2) @ (x @ u)
        g = self.x_star
        gt = g.transpose(0, 1)
        if kind == "diff_left":
            return x @ (x.transpose(1, 2) @ u) - g @ (gt @ u)
        return x.transpose(1, 2) @ (x @ u) - gt @ (g @ u)

    def extract_representation(self, x: torch.Tensor) -> torch.Tensor:
        response = self.raw_probe_response(x)
        if self.config.standardize:
            response = _standardize(response)
        return self.encoder(_project(response, self.projection))


class MultiViewProbe(_PredictionModel):
    """Four-view MVProbe/GramDiff and independent six/eight-view controls."""

    def __init__(self, config: ModelConfig, reference: torch.Tensor | None) -> None:
        super().__init__(config)
        d_l, d_r = self.input_shape
        # Preserve module creation and probe-reset order, including RNG draws.
        self.probes_xu = nn.Linear(config.n_probes, d_r, bias=False)
        self.probes_xxtu = nn.Linear(config.n_probes, d_l, bias=False)
        self.probes_xtu = nn.Linear(config.n_probes, d_l, bias=False)
        self.probes_xtxu = nn.Linear(config.n_probes, d_r, bias=False)
        self._initialize_probes((self.probes_xu, self.probes_xxtu,
                                 self.probes_xtu, self.probes_xtxu))
        self.shared_probe_proj_row = nn.Linear(d_l, config.proj_dim, bias=False)
        self.shared_probe_proj_row2 = nn.Linear(d_l, config.proj_dim, bias=False)
        self.shared_probe_proj_col = nn.Linear(d_r, config.proj_dim, bias=False)
        self.shared_probe_proj_col2 = nn.Linear(d_r, config.proj_dim, bias=False)
        degrees = {
            "mvprobe": (), "gramdiff": (), "poly6": (3,), "poly8": (3, 4),
            "duplicate6": (1,), "duplicate8": (1, 2),
        }[config.family]
        self.extra_specs = tuple((side, degree) for degree in degrees for side in ("left", "right"))
        self.extra_probes = nn.ModuleList(
            nn.Linear(config.n_probes,
                      (d_r if side == "left" else d_l) if degree % 2
                      else (d_l if side == "left" else d_r), bias=False)
            for side, degree in self.extra_specs
        )
        self._initialize_probes(self.extra_probes)
        self.extra_projections = nn.ModuleList(
            nn.Linear(d_l if side == "left" else d_r, config.proj_dim, bias=False)
            for side, _ in self.extra_specs
        )
        self.concat_all4_encoder = nn.Linear(
            (4 + len(self.extra_specs)) * config.proj_dim * config.n_probes, config.rep_dim)
        self._create_head()
        if config.family == "gramdiff":
            if reference is None:
                raise ValueError("gramdiff requires a reference")
            self._register_reference(reference)
        elif reference is not None:
            raise ValueError("only gramdiff and single families use a reference")

    def raw_probe_responses(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(x)
        xt = x.transpose(1, 2)
        if self.config.family == "gramdiff":
            # Gram differences are evaluated as written: X(X^T U) - G(G^T U),
            # never as the Gram of X - G. The reference products do not depend
            # on the batch, so they are computed once per forward.
            g = self.x_star
            gt = g.transpose(0, 1)
            return {
                "xu": x @ self.probes_xu.weight - g @ self.probes_xu.weight,
                "xxtu": x @ (xt @ self.probes_xxtu.weight)
                - g @ (gt @ self.probes_xxtu.weight),
                "xtu": xt @ self.probes_xtu.weight - gt @ self.probes_xtu.weight,
                "xtxu": xt @ (x @ self.probes_xtxu.weight)
                - gt @ (g @ self.probes_xtxu.weight),
            }
        if self.config.family == "mvprobe":
            return {
                "xu": x @ self.probes_xu.weight,
                "xxtu": x @ (x.transpose(1, 2) @ self.probes_xxtu.weight),
                "xtu": xt @ self.probes_xtu.weight,
                "xtxu": xt @ (xt.transpose(1, 2) @ self.probes_xtxu.weight),
            }
        responses = {
            "xu": x @ self.probes_xu.weight,
            "xxtu": x @ (xt @ self.probes_xxtu.weight),
            "xtu": xt @ self.probes_xtu.weight,
            "xtxu": xt @ (x @ self.probes_xtxu.weight),
        }
        for index, ((side, degree), probe) in enumerate(zip(self.extra_specs, self.extra_probes)):
            response = probe.weight
            for factor_index in reversed(range(degree)):
                factor = x if (factor_index % 2 == 0) == (side == "left") else xt
                response = factor @ response
            responses[f"extra_{index}"] = response
        return responses

    def extract_representation(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.family == "mvprobe" and self.config.task == "membership":
            # Interleave raw responses and standardization as in four-view
            # MVProbe; this also preserves floating-point gradient ordering.
            self._validate_input(x)
            xt = x.transpose(1, 2)
            normalize = _standardize if self.config.standardize else lambda value: value
            responses = (
                normalize(x @ self.probes_xu.weight),
                normalize(x @ (x.transpose(1, 2) @ self.probes_xxtu.weight)),
                normalize(xt @ self.probes_xtu.weight),
                normalize(xt @ (xt.transpose(1, 2) @ self.probes_xtxu.weight)),
            )
        else:
            responses = tuple(self.raw_probe_responses(x).values())
            if self.config.standardize:
                responses = tuple(_standardize(response) for response in responses)
        projections = (self.shared_probe_proj_row, self.shared_probe_proj_row2,
                       self.shared_probe_proj_col, self.shared_probe_proj_col2,
                       *self.extra_projections)
        features = [_project(response, projection) for response, projection in zip(responses, projections)]
        return self.concat_all4_encoder(torch.cat(features, dim=-1))


def build_model(
    config: ModelConfig,
    reference: torch.Tensor | None = None,
) -> nn.Module:
    """Build a standalone model; reference provenance is handled by the caller."""
    if config.reference_trainable and reference is None:
        raise ValueError("reference_trainable requires a reference")
    if config.family == "probex":
        if reference is not None:
            raise ValueError("probex does not use a reference")
        return ProbeX(config)
    if config.family == "single":
        return SingleBranch(config, reference)
    return MultiViewProbe(config, reference)


__all__ = ["ModelConfig", "build_model"]
