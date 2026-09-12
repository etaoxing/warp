# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code-generation bookkeeping for launch-bound ``tile_gemm`` pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def complete_gemm_pipeline_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Add the array shapes and element strides used by launch eligibility checks."""
    array_fields = ("a_shape", "b_shape", "a_element_strides", "b_element_strides")
    if all(field in metadata for field in array_fields):
        return metadata

    full_m, full_n, full_k = metadata["shape"]
    a_layout, b_layout, _ = metadata["arrangements"]
    a_shape = (full_m, full_k) if a_layout == "rowmajor" else (full_k, full_m)
    b_shape = (full_k, full_n) if b_layout == "rowmajor" else (full_n, full_k)
    metadata.update(
        a_shape=a_shape,
        b_shape=b_shape,
        a_element_strides=(a_shape[1], 1),
        b_element_strides=(b_shape[1], 1),
    )
    return metadata


@dataclass
class GemmPipelineRequirement:
    """One hidden device-pipeline argument required by a generated function."""

    name: str
    pipeline: dict[str, Any]
    a_label: str
    b_label: str


class GemmPipelineCodegen:
    """Thread ``tile_gemm`` device pipelines through kernels and ``@wp.func`` calls."""

    def __init__(self, adj):
        self.adj = adj
        adj.gemm_pipeline_requirements = []
        self._requirements = {}
        self.has_unbound_pipeline = False
        self._calls = []

    def begin_build(self) -> None:
        self.adj.gemm_pipeline_requirements = []
        self._requirements = {}
        self.has_unbound_pipeline = False
        self._calls = []

    @staticmethod
    def _direct_array_arg(adj, value):
        """Return a direct array argument, or ``None`` for device-created views."""
        if value is None or getattr(value, "parent", None) is not None:
            return None
        label = getattr(value, "label", None)
        if label is None:
            return None
        return value if any(arg.label == label for arg in adj.args) else None

    def _add(self, pipeline, a, b):
        a = self._direct_array_arg(self.adj, a)
        b = self._direct_array_arg(self.adj, b)
        if a is None or b is None:
            return None

        key = (pipeline["lto_symbol"], a.label, b.label)
        requirement = self._requirements.get(key)
        if requirement is None:
            requirement = GemmPipelineRequirement(
                name=f"_wp_gemm_pipeline_{len(self._requirements)}",
                pipeline=pipeline,
                a_label=a.label,
                b_label=b.label,
            )
            self._requirements[key] = requirement
            self.adj.gemm_pipeline_requirements.append(requirement)
        return requirement

    def bind_builtin(self, func, func_args, bound_args):
        """Replace a pipeline placeholder in a direct ``tile_gemm`` call."""
        if not (func.is_builtin() and func.key == "tile_gemm"):
            return func_args

        from warp._src.codegen import Var  # noqa: PLC0415

        bound = list(func_args)
        for index, arg in enumerate(bound):
            pipeline = getattr(arg, "gemm_pipeline", None)
            if pipeline is None:
                continue
            requirement = self._add(pipeline, bound_args.get("a"), bound_args.get("b"))
            if requirement is None:
                self.has_unbound_pipeline = True
            label = requirement.name if requirement is not None else "0"
            bound[index] = Var(label, str, False, True, False)
        return tuple(bound)

    @staticmethod
    def _mapped_args(adj, requirement, bound_args):
        return (
            GemmPipelineCodegen._direct_array_arg(adj, bound_args.get(requirement.a_label)),
            GemmPipelineCodegen._direct_array_arg(adj, bound_args.get(requirement.b_label)),
        )

    def helper_args_for(self, func, bound_args):
        if func is None or self.adj.builder_options.get("output_arch") is None:
            return []
        # Custom gradients/replays are built after their callers. Delay argument
        # expansion until their requirements (and those of nested callers) exist.
        if self.adj.builder is not None:
            token = f"__wp_gemm_call_{len(self._calls)}__"
            self._calls.append((token, func, dict(bound_args)))
            return [token]
        return self._bound_helper_args(func, bound_args)

    def _bound_helper_args(self, func, bound_args):
        if func.adj.gemm_pipeline.has_unbound_pipeline:
            self.has_unbound_pipeline = True

        args = []
        for requirement in getattr(func.adj, "gemm_pipeline_requirements", ()):
            a, b = self._mapped_args(self.adj, requirement, bound_args)
            mapped = self._add(requirement.pipeline, a, b) if a is not None and b is not None else None
            if mapped is None:
                self.has_unbound_pipeline = True
            args.append(mapped.name if mapped is not None else "0")
        return args

    @staticmethod
    def finalize(adjs):
        """Resolve hidden arguments after every deferred helper has been built."""
        generators = [adj.gemm_pipeline for adj in adjs]
        if not any(gen._calls for gen in generators):
            return

        def state():
            return tuple((len(gen._requirements), gen.has_unbound_pipeline) for gen in generators)

        previous = None
        while previous != state():
            previous = state()
            for gen in generators:
                for _, func, bound_args in gen._calls:
                    gen._bound_helper_args(func, bound_args)

        for gen in generators:
            if not gen._calls:
                continue
            replacements = [
                (token, ", ".join(gen._bound_helper_args(func, bound_args))) for token, func, bound_args in gen._calls
            ]
            block = gen.adj.blocks[0]
            for body in (block.body_forward, block.body_replay, block.body_reverse):
                for index in range(len(body)):
                    line = body[index]
                    for token, arguments in replacements:
                        if not arguments:
                            line = line.replace(f", {token}", "").replace(f"{token}, ", "")
                        line = line.replace(token, arguments)
                    body[index] = line

    def call_args(self, func, bound_args):
        if func.is_builtin():
            return []
        return self.helper_args_for(func, bound_args)

    def replay_call_args(self, func, bound_args, default_args):
        if func.custom_replay_func is None:
            return default_args
        return self.helper_args_for(func.custom_replay_func, bound_args)

    def reverse_call_args(self, func, bound_args, default_args):
        if func.custom_grad_func is None:
            return default_args
        return self.helper_args_for(func.custom_grad_func, bound_args)

    def function_args(self):
        return [f"void* {requirement.name}" for requirement in self.adj.gemm_pipeline_requirements]

    def kernel_args(self):
        return self.function_args()

    def kernel_metadata(self):
        arg_indices = {arg.label: index for index, arg in enumerate(self.adj.args)}

        def metadata(requirement):
            pipeline = requirement.pipeline
            return complete_gemm_pipeline_metadata(
                {
                    "name": requirement.name,
                    "a_arg": arg_indices[requirement.a_label],
                    "b_arg": arg_indices[requirement.b_label],
                    **pipeline,
                }
            )

        return [metadata(requirement) for requirement in self.adj.gemm_pipeline_requirements]
