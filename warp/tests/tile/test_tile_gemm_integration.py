# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host and code-generation regressions for launch-bound GEMM pipelines."""

import contextlib
import ctypes
import json
import re
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import warp as wp
from warp._src import build, builtins, codegen, context, deterministic
from warp._src.gemm_pipeline import GemmPipelineCodegen, complete_gemm_pipeline_metadata


def _make_custom_kernel(replay):
    @wp.func
    def product(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
        result = wp.tile_load(a, shape=(16, 16))
        wp.tile_store(c, result)

    if replay:

        @wp.func_replay(product)
        def replay_product(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
            result = wp.tile_gemm(a, b, 0, 0, tile=(16, 16, 16), shape=(32, 32, 32))
            wp.tile_store(c, result)

        custom = product.custom_replay_func
    else:

        @wp.func_grad(product)
        def adj_product(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
            result = wp.tile_gemm(a, b, 0, 0, tile=(16, 16, 16), shape=(32, 32, 32))
            wp.tile_store(wp.adjoint[c], result)

        custom = product.custom_grad_func

    @wp.func
    def nested(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
        product(a, b, c)

    @wp.kernel(module=f"gemm_integration_{uuid.uuid4().hex}")
    def kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float]):
        nested(a, b, c)

    return kernel, nested, custom


def _pipeline_metadata():
    return complete_gemm_pipeline_metadata(
        {
            "lto_symbol": "integration_gemm",
            "lto": b"",
            "symbols": {name: f"integration_{name}" for name in wp._src.build._GEMM_PIPELINE_SYMBOL_SUFFIXES},
            "init_kernel": "integration_init",
            "destroy_kernel": "integration_destroy",
            "tile": (16, 16, 16),
            "shape": (32, 32, 32),
            "arrangements": ("rowmajor", "rowmajor", "rowmajor"),
            "device_pipeline_storage_size": 32,
            "device_pipeline_storage_alignment": 16,
            "shared_memory_buffer_size": 4096,
            "shared_memory_buffer_alignment": 16,
            "tile_pipeline_storage_size": 32,
            "tile_pipeline_storage_alignment": 16,
            "accumulator_storage_size": 32,
            "accumulator_storage_alignment": 16,
            "pipeline_block_dim": 64,
            "a_arg": 0,
            "b_arg": 1,
        }
    )


def _make_execution_kernel(replay):
    @wp.func
    def product(a: wp.array2d[float], b: wp.array2d[float], g: wp.array2d[float], c: wp.array2d[float]):
        result = wp.tile_gemm(a, b, 0, 0, tile=(32, 32, 16), shape=(32, 32, 64))
        wp.tile_store(c, result)

    if replay:

        @wp.func_replay(product)
        def replay_product(a: wp.array2d[float], b: wp.array2d[float], g: wp.array2d[float], c: wp.array2d[float]):
            # Keep a live tile across a pipeline with a larger K tile so this
            # deferred frame exceeds both the primal and its automatic adjoint.
            live = wp.tile_ones(shape=(64, 64), dtype=float, storage="shared")
            result = wp.tile_gemm(a, b, 0, 0, tile=(32, 32, 32), shape=(32, 32, 64))
            wp.tile_store(c, result * live[0, 0])

        custom = product.custom_replay_func
    else:

        @wp.func_grad(product)
        def adj_product(a: wp.array2d[float], b: wp.array2d[float], g: wp.array2d[float], c: wp.array2d[float]):
            # The caller also supplies g as c's upstream gradient. Keeping it a
            # global argument makes both custom-gradient pipelines launch-bound.
            grad_a = wp.tile_gemm(g, b, 0, 0, tile=(32, 64, 16), shape=(32, 64, 32), transpose_b=True)
            grad_b = wp.tile_gemm(a, g, 0, 0, tile=(64, 32, 16), shape=(64, 32, 32), transpose_a=True)
            wp.tile_store(wp.adjoint[a], grad_a)
            wp.tile_store(wp.adjoint[b], grad_b)

        custom = product.custom_grad_func

    @wp.func
    def nested(a: wp.array2d[float], b: wp.array2d[float], g: wp.array2d[float], c: wp.array2d[float]):
        product(a, b, g, c)

    @wp.kernel(module=f"gemm_execution_{uuid.uuid4().hex}")
    def kernel(a: wp.array2d[float], b: wp.array2d[float], g: wp.array2d[float], c: wp.array2d[float]):
        nested(a, b, g, c)

    return kernel, custom


def _make_workspace_kernel():
    @wp.kernel(
        module=f"gemm_workspace_{uuid.uuid4().hex}",
        enable_backward=False,
    )
    def kernel(a: wp.array2d[float], b: wp.array2d[float], c: wp.array2d[float], scratch: wp.array2d[float]):
        live = wp.tile_ones(shape=(128, 64), dtype=float, storage="shared")
        result = wp.tile_gemm(a, b, 0, 0, tile=(128, 128, 32), shape=(128, 128, 64))
        wp.tile_store(c, result)
        wp.tile_store(scratch, live)

    wp.set_module_options({"enable_backward": False}, module=kernel.module)
    return kernel


def _mock_pipeline_metadata(*args):
    metadata = _pipeline_metadata()
    symbol = "gemm_" + "_".join(map(str, args[:6])) + "_" + "_".join(args[9:12])
    metadata.update(
        lto_symbol=symbol,
        tile=args[:3],
        shape=args[3:6],
        arrangements=args[9:12],
        shared_memory_buffer_size=2 * args[2] * (args[0] + args[1]) * 4 + 32,
    )
    for field in ("a_shape", "b_shape", "a_element_strides", "b_element_strides"):
        del metadata[field]
    return complete_gemm_pipeline_metadata(metadata)


class _Array:
    def __init__(self, device, ptr):
        self.device = device
        self.ptr = ptr
        self.dtype = wp.float32
        self.shape = (32, 32)
        self.strides = (128, 4)
        self.ndim = 2


class TestTileGemmIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_gemm_residency_bounds(self):
        adj = SimpleNamespace(get_total_required_shared_gemm_pipeline=lambda: 16384)
        pipeline = GemmPipelineCodegen(adj)
        adj.gemm_pipeline_requirements.append(object())
        options = {
            "_gemm_launch_limits_v1": [102400, 1024, 1536, 24, 32, 128],
            "output_arch": 120,
            "device_arch": 120,
            "block_dim": 64,
        }
        self.assertEqual(pipeline.forward_launch_bounds(options), "__launch_bounds__(64, 5) ")
        # This boundary distinguishes CUDA's 128-byte allocation unit from 256.
        adj.get_total_required_shared_gemm_pipeline = lambda: 15984
        self.assertEqual(pipeline.forward_launch_bounds(options), "__launch_bounds__(64, 6) ")
        adj.get_total_required_shared_gemm_pipeline = lambda: 65584
        self.assertEqual(pipeline.forward_launch_bounds(options | {"block_dim": 256}), "")
        adj.get_total_required_shared_gemm_pipeline = lambda: 16384
        for override in (
            {"_gemm_launch_limits_v1": None},
            {"output_arch": 89},
            {"llvm_cuda": True},
            {"cluster_dim": 2},
            {"launch_bounds": (64, 2)},
            {"cuda_max_registers": 100},
        ):
            with self.subTest(override=override):
                self.assertEqual(pipeline.forward_launch_bounds(options | override), "")
        pipeline.has_unbound_pipeline = True
        self.assertEqual(pipeline.forward_launch_bounds(options), "")
        pipeline.has_unbound_pipeline = False
        adj.gemm_pipeline_requirements.clear()
        self.assertEqual(pipeline.forward_launch_bounds(options), "")

    def test_gemm_residency_bound_is_forward_only(self):
        kernel, _ = _make_execution_kernel(False)
        kernel.module.get_module_hash(64)
        options = kernel.module.resolve_options(wp.config, block_dim=64) | {
            "output_arch": 120,
            "device_arch": 120,
            "_gemm_launch_limits_v1": [102400, 1024, 1536, 24, 32, 128],
        }

        def compile_pipeline(*args):
            metadata = _mock_pipeline_metadata(*args)
            args[-1].gemm_pipeline_metadata[metadata["lto_symbol"]] = metadata
            return metadata

        with (
            mock.patch.object(builtins, "_tile_gemm_pipeline_supported", return_value=True),
            mock.patch.object(build, "build_lto_gemm_pipeline", side_effect=compile_pipeline),
        ):
            for explicit in ({}, {"launch_bounds": (64, 2)}, {"cuda_max_registers": 100}):
                source, _, meta, _, _ = kernel.module._run_codegen(options | explicit, is_cpu=False)
                forward = next(line for line in source.splitlines() if codegen.cuda_kernel_forward_name(kernel) in line)
                backward = next(
                    line for line in source.splitlines() if codegen.cuda_kernel_backward_name(kernel) in line
                )
                self.assertEqual("__launch_bounds__" in forward, "cuda_max_registers" not in explicit)
                self.assertEqual("__launch_bounds__" in backward, "launch_bounds" in explicit)
                if "launch_bounds" in explicit:
                    self.assertIn("__launch_bounds__(64, 2)", forward)
                    self.assertIn("__launch_bounds__(64, 2)", backward)
                self.assertEqual(meta["_gemm_launch_limits_v1"], options["_gemm_launch_limits_v1"])

    def test_pipeline_precision_abi_and_cache(self):
        calls = []
        metadata = _pipeline_metadata()
        compiler = context.runtime.core.wp_cuda_compile_gemm_pipeline

        @ctypes.CFUNCTYPE(ctypes.c_bool, *compiler.argtypes)
        def compile_pipeline(*args):
            calls.append(args[9:21])
            for field, output in zip(build._GEMM_PIPELINE_METADATA_FIELDS, args[21:], strict=True):
                output[0] = metadata[field]
            Path(args[0].decode()).write_bytes(args[1])
            return True

        def compile_types(dtypes):
            builder = SimpleNamespace(ltoirs={}, ltoirs_decl={}, native_code={}, gemm_pipeline_metadata={})
            return build.build_lto_gemm_pipeline(
                16, 16, 16, 32, 32, 32, *dtypes, "rowmajor", "colmajor", "rowmajor", 120, 64, 2, builder
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(wp.config, "kernel_cache_dir", directory),
            mock.patch.object(context.runtime.core, "wp_cuda_compile_gemm_pipeline", compile_pipeline),
        ):
            symbols = []
            for dtypes, precision_key in (
                ((wp.float32, wp.float32, wp.float32), "5_5_5"),
                ((wp.float16, wp.float32, wp.float64), "3_5_6"),
            ):
                result = compile_types(dtypes)
                symbol = result["lto_symbol"]
                self.assertTrue(symbol.startswith("gemm_pipeline_v8_"))
                self.assertTrue(symbol.endswith(f"_storage_{precision_key}_arithmetic_{precision_key}_0"))
                self.assertEqual(result["lto"], symbol.encode())
                # A new builder must reuse the persistent entry for the same
                # storage/arithmetic contract without calling the compiler again.
                self.assertEqual(compile_types(dtypes)["lto"], result["lto"])
                symbols.append(symbol)

        self.assertNotEqual(*symbols)
        self.assertEqual(
            calls,
            [
                (5, 5, 5, 5, 5, 5, 0, 1, 0, 1, 64, 2),
                (3, 5, 6, 3, 5, 6, 0, 1, 0, 1, 64, 2),
            ],
        )

    def test_deferred_custom_pipeline_arguments(self):
        for replay in (False, True):
            with self.subTest(replay=replay):
                kernel, nested, custom = _make_custom_kernel(replay)
                kernel.module.get_module_hash(64)
                self.assertFalse(custom.adj.gemm_pipeline_requirements)
                options = kernel.module.resolve_options(wp.config, block_dim=64)
                options |= {"output_arch": 120, "device_arch": 120}
                with (
                    mock.patch.object(builtins, "_tile_gemm_pipeline_supported", return_value=True),
                    mock.patch.object(wp._src.build, "build_lto_gemm_pipeline", return_value=_pipeline_metadata()),
                ):
                    source, _, meta, _, _ = kernel.module._run_codegen(options, is_cpu=False)

                self.assertNotIn("__wp_gemm_call_", source)
                self.assertEqual(len(custom.adj.gemm_pipeline_requirements), 1)
                self.assertEqual(len(nested.adj.gemm_pipeline_requirements), 1)
                self.assertEqual(len(kernel.adj.gemm_pipeline_requirements), 1)
                for adj in (nested.adj, kernel.adj):
                    body = "\n".join(adj.blocks[0].body_replay + adj.blocks[0].body_reverse)
                    self.assertIn("_wp_gemm_pipeline_0", body)
                requirements = meta[codegen.cuda_kernel_forward_name(kernel) + "_gemm_pipelines"]
                self.assertEqual((requirements[0]["a_arg"], requirements[0]["b_arg"]), (0, 1))

                # The same Python helper objects are reused when another backend
                # builds this module; deferred CUDA requirements must not leak in.
                cpu_options = options | {"output_arch": None, "device_arch": None}
                cpu_source, _, _, _, _ = kernel.module._run_codegen(cpu_options, is_cpu=True)
                self.assertNotIn("_wp_gemm_pipeline_", cpu_source)
                self.assertNotIn("__wp_gemm_call_", cpu_source)

    def test_cuda_metadata_does_not_promote_other_architectures(self):
        kernel, _, _ = _make_custom_kernel(False)
        module = kernel.module
        options = module.resolve_options(wp.config, block_dim=64)

        def device(arch):
            return SimpleNamespace(
                arch=arch,
                is_cuda=True,
                is_cpu=False,
                context=arch,
                alias=f"sm{arch}",
                _get_cuda_arch_suffix=lambda _: "",
                _gemm_launch_limits=(102400, 1024, 1536, 24, 32, 128) if arch == 120 else None,
            )

        def generate(options, is_cpu):
            meta = {"test_smem_bytes": options["output_arch"]}
            if options["output_arch"] == 120:
                meta["_cuda_arch_suffix"] = "a"
                meta["_gemm_launch_limits_v1"] = options["_gemm_launch_limits_v1"]
            return "// mocked device code", "cu", meta, [], []

        def compile_cuda(source_path, arch, output_path, **kwargs):
            Path(output_path).write_bytes(b"test binary")

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(wp.config, "cuda_arch_suffix", None),
            mock.patch.object(context.runtime, "toolkit_version", (13, 2)),
            mock.patch.object(context.runtime, "get_nvrtc_pch_dir", return_value=None),
            mock.patch.object(module, "_run_codegen", side_effect=generate) as codegen,
            mock.patch.object(wp._src.build, "build_cuda", side_effect=compile_cuda),
        ):
            legacy_name = module._get_compile_output_name(None, 120, "a", False, 64)
            legacy_binary = Path(directory) / legacy_name
            legacy_binary.write_bytes(b"legacy binary")
            (Path(directory) / module._get_meta_name(64, legacy_name)).write_text(
                json.dumps({"_cuda_arch_suffix": "a"})
            )
            compiled, name120, _ = module._compile(
                device(120), directory, output_arch=120, use_ptx=False, options=options
            )
            self.assertTrue(compiled)
            self.assertIn(".sm120a.", name120)
            self.assertNotEqual(name120, legacy_name)
            self.assertEqual(legacy_binary.read_bytes(), b"legacy binary")
            # Exercise a discovery manifest left by the previous cache schema.
            manifest = Path(directory) / module._get_meta_name(64)
            manifest.write_text(json.dumps({"_cuda_arch_suffix": "a", "_cuda_binary_metadata": True}))

            for arch in (89, 90):
                compiled, name, _ = module._compile(
                    device(arch), directory, output_arch=arch, use_ptx=False, options=options
                )
                self.assertTrue(compiled)
                self.assertIn(f".sm{arch}.", name)
                metadata = json.loads((Path(directory) / module._get_meta_name(64, name)).read_text())
                self.assertEqual(metadata["test_smem_bytes"], arch)

            codegen.reset_mock()
            compiled, name, _ = module._compile(device(120), directory, output_arch=120, use_ptx=False, options=options)
            self.assertFalse(compiled)
            self.assertEqual(name, name120)
            codegen.assert_not_called()

            changed_device = device(120)
            changed_device._gemm_launch_limits = (102400, 2048, 1536, 24, 32, 128)
            compiled, changed_name, _ = module._compile(
                changed_device, directory, output_arch=120, use_ptx=False, options=options
            )
            self.assertTrue(compiled, "Changed device limits must invalidate the compiled residency policy")
            codegen.assert_called_once()
            self.assertNotEqual(changed_name, name120)
            self.assertTrue((Path(directory) / name120).exists())
            metadata_path = Path(directory) / module._get_meta_name(64, changed_name)
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["_gemm_launch_limits_v1"], list(changed_device._gemm_launch_limits))
            codegen.reset_mock()
            compiled, _, _ = module._compile(changed_device, directory, output_arch=120, use_ptx=False, options=options)
            self.assertFalse(compiled)
            codegen.assert_not_called()
            self.assertIsNone(context._get_gemm_launch_limits(changed_device, 89))

            with (
                mock.patch.dict(module.options, {"block_dim": 64}),
                mock.patch.object(context, "get_device", return_value=changed_device),
                mock.patch.object(context, "is_cuda_driver_initialized", return_value=False),
                mock.patch.object(context, "is_cuda_available", return_value=False),
                mock.patch.object(module, "load") as load,
            ):
                context.load_aot_module(module, device=changed_device, arch=120, module_dir=directory, use_ptx=False)
                self.assertEqual(Path(load.call_args.kwargs["binary_path"]).name, changed_name)
                changed_device._gemm_launch_limits = (102400, 4096, 1536, 24, 32, 128)
                context.load_aot_module(module, device=changed_device, arch=120, module_dir=directory, use_ptx=False)
                self.assertEqual(Path(load.call_args.kwargs["binary_path"]).name, legacy_name)
                legacy_binary.unlink()
                context.load_aot_module(module, device=changed_device, arch=120, module_dir=directory, use_ptx=False)
                self.assertIn(Path(load.call_args.kwargs["binary_path"]).name, (name120, changed_name))

    def test_aot_preserves_explicit_suffix_without_gemm_metadata(self):
        @wp.kernel(module="unique", enable_backward=False)
        def kernel():
            pass

        module = kernel.module

        def device(reserved):
            return SimpleNamespace(
                arch=120,
                is_cuda=True,
                is_cpu=False,
                alias="sm120",
                _gemm_launch_limits=(102400, reserved, 1536, 24, 32, 128),
                _get_cuda_arch_suffix=lambda _: "a",
            )

        target = device(2048)
        for export_device, use_ptx in ((None, False), (None, True), (device(1024), False), (device(1024), True)):
            with (
                self.subTest(device_specific=export_device is not None, use_ptx=use_ptx),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.object(wp.config, "cuda_arch_suffix", "a"),
                mock.patch.object(context, "get_device", return_value=target),
                mock.patch.object(context, "is_cuda_driver_initialized", return_value=False),
                mock.patch.object(context, "is_cuda_available", return_value=False),
                mock.patch.object(module, "load") as load,
            ):
                name = module._get_compile_output_name(export_device, 120, "a", use_ptx)
                binary = Path(directory) / name
                binary.write_bytes(b"AOT binary")
                meta = Path(directory) / module._get_meta_name(output_name=name)
                # Ordinary kernels have no linked-GEMM architecture requirement.
                meta.write_text(json.dumps({"_cuda_binary_metadata": True}))
                context.load_aot_module(module, device=target, arch=120, module_dir=directory, use_ptx=use_ptx)
                self.assertEqual(Path(load.call_args.kwargs["binary_path"]), binary)
                self.assertEqual(Path(load.call_args.kwargs["meta_path"]), meta)

    def test_deferred_helpers_reserve_their_workspace(self):
        for replay in (False, True):
            with self.subTest(replay=replay):
                kernel, custom = _make_execution_kernel(replay)
                kernel.module.get_module_hash(64)
                options = kernel.module.resolve_options(wp.config, block_dim=64)
                options |= {"output_arch": 120, "device_arch": 120}
                with (
                    mock.patch.object(builtins, "_tile_gemm_pipeline_supported", return_value=True),
                    mock.patch.object(wp._src.build, "build_lto_gemm_pipeline", side_effect=_mock_pipeline_metadata),
                ):
                    _, _, meta, _, _ = kernel.module._run_codegen(options, is_cpu=False)
                forward = meta[codegen.cuda_kernel_forward_name(kernel) + "_smem_bytes"]
                backward = meta[codegen.cuda_kernel_backward_name(kernel) + "_smem_bytes"]
                self.assertGreater(custom.adj.get_total_required_shared(), forward)
                self.assertGreaterEqual(backward, custom.adj.get_total_required_shared())

    def test_redefined_custom_replay_symbol(self):
        # A prior definition without a replay makes the primal and replay name
        # counters differ, as they do when a closure is recreated in user code.
        _make_execution_kernel(False)
        kernel, replay = _make_execution_kernel(True)
        kernel.module.get_module_hash(64)
        options = kernel.module.resolve_options(wp.config, block_dim=64)
        options |= {"output_arch": 120, "device_arch": 120}
        with (
            mock.patch.object(builtins, "_tile_gemm_pipeline_supported", return_value=True),
            mock.patch.object(build, "build_lto_gemm_pipeline", side_effect=_mock_pipeline_metadata),
        ):
            source, _, _, _, _ = kernel.module._run_codegen(options, is_cpu=False)

        declaration = re.search(rf"static CUDA_CALLABLE void {re.escape(replay.native_func)}\(", source)
        call = re.search(rf"(?m)^\s*{re.escape(replay.native_func)}\(", source)
        self.assertIsNotNone(declaration, "The registered custom replay must be defined")
        self.assertIsNotNone(call, "The caller must use the registered replay's native identifier")
        self.assertLess(declaration.start(), call.start())

    def test_pipeline_workspace_fits_without_portable_workspace(self):
        kernel = _make_workspace_kernel()
        kernel.module.get_module_hash(256)
        options = kernel.module.resolve_options(wp.config, block_dim=256)
        options |= {"output_arch": 120, "device_arch": 120}
        with (
            mock.patch.object(builtins, "_tile_gemm_pipeline_supported", return_value=True),
            mock.patch.object(wp._src.build, "build_lto_gemm_pipeline", side_effect=_mock_pipeline_metadata),
        ):
            _, _, meta, _, _ = kernel.module._run_codegen(options, is_cpu=False)
        name = codegen.cuda_kernel_forward_name(kernel)
        full = meta[name + "_smem_bytes"]
        pipeline = meta[name + "_gemm_pipeline_smem_bytes"]
        device = SimpleNamespace(is_cuda=True, context=None, alias="sm120", max_shared_memory_per_block=101376)
        self.assertLess(pipeline + 32, device.max_shared_memory_per_block)
        self.assertGreater(full, device.max_shared_memory_per_block)

        with (
            mock.patch.object(context.runtime.core, "wp_cuda_get_kernel", return_value=1),
            mock.patch.object(context.runtime.core, "wp_cuda_get_kernel_static_shared_memory", return_value=32),
            mock.patch.object(
                context.runtime.core,
                "wp_cuda_configure_kernel_shared_memory",
                side_effect=lambda function, size: size + 32 <= device.max_shared_memory_per_block,
            ) as configure,
            mock.patch.object(context, "log_warning"),
        ):
            module = context.ModuleExec(None, b"test", device, meta, 256, 120)
            hooks = module.get_kernel_hooks(kernel)
            self.assertEqual([call.args[1] for call in configure.call_args_list], [full, pipeline])
            self.assertTrue(hooks.gemm_pipeline_enabled)
            self.assertEqual(hooks.select_forward_smem_bytes([ctypes.c_void_p(1)]), pipeline)
            for pointers in ([ctypes.c_void_p()], [ctypes.c_void_p(1), ctypes.c_void_p()]):
                with self.assertRaisesRegex(RuntimeError, "portable or mixed launch needs"):
                    hooks.select_forward_smem_bytes(pointers)

            # An unbound helper keeps the conservative pipeline roofline. It
            # must not accidentally inherit the smaller allocation above.
            unbound_meta = meta | {name + "_gemm_pipeline_smem_bytes": full}
            unbound = context.ModuleExec(None, b"test", device, unbound_meta, 256, 120).get_kernel_hooks(kernel)
            self.assertEqual(unbound.select_forward_smem_bytes([ctypes.c_void_p(1)]), full)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_pipeline_only_workspace_execution(self):
        device = wp.get_device("cuda:0")
        if (
            device.arch < 100
            or (context.runtime.toolkit_version or (0, 0)) < (13, 1)
            or not context.runtime.core.wp_is_mathdx_enabled()
        ):
            self.skipTest("Requires supported GEMM pipelines")
        kernel = _make_workspace_kernel()
        module = kernel.module.load(device, 256)
        hooks = module.get_kernel_hooks(kernel)
        if not (hooks.gemm_pipeline_smem_bytes < device.max_shared_memory_per_block < hooks.forward_smem_bytes):
            self.skipTest("The generated workspaces do not straddle this device's shared-memory limit")
        self.assertTrue(hooks.gemm_pipeline_enabled)

        rng = np.random.default_rng(59)
        a_host = rng.standard_normal((128, 64), dtype=np.float32)
        b_host = rng.standard_normal((64, 128), dtype=np.float32)
        a = wp.array(a_host, device=device)
        b = wp.array(b_host, device=device)
        c = wp.empty((128, 128), device=device)
        scratch = wp.empty((128, 64), device=device)
        command = wp.launch_tiled(
            kernel, dim=1, inputs=[a, b], outputs=[c, scratch], block_dim=256, device=device, record_cmd=True
        )
        command.launch()
        self.assertEqual(command._forward_smem_bytes, hooks.gemm_pipeline_smem_bytes)
        np.testing.assert_allclose(c.numpy(), a_host @ b_host, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(scratch.numpy(), np.ones((128, 64), dtype=np.float32))
        with wp.ScopedCapture(device, force_module_load=False) as capture:
            command.launch()
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(c.numpy(), a_host @ b_host, rtol=1e-4, atol=1e-4)

        padded = np.zeros((128, 65), dtype=np.float32)
        padded[:, 1:] = a_host
        sliced = wp.array(padded, device=device)[:, 1:]
        command.set_param_by_name("a", sliced)
        with self.assertRaisesRegex(RuntimeError, "portable or mixed launch needs"):
            command.launch()
        with self.assertRaisesRegex(RuntimeError, "portable or mixed launch needs"):
            wp.launch_tiled(kernel, dim=1, inputs=[sliced, b], outputs=[c, scratch], block_dim=256, device=device)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_custom_pipeline_execution_and_workspace(self):
        device = wp.get_device("cuda:0")
        if (
            device.arch < 100
            or (context.runtime.toolkit_version or (0, 0)) < (13, 1)
            or not context.runtime.core.wp_is_mathdx_enabled()
        ):
            self.skipTest("Requires supported GEMM pipelines")

        rng = np.random.default_rng(47)
        a_host = rng.standard_normal((32, 64), dtype=np.float32)
        b_host = rng.standard_normal((64, 32), dtype=np.float32)
        g_host = rng.standard_normal((32, 32), dtype=np.float32)
        for replay in (False, True):
            with self.subTest(replay=replay):
                kernel, custom = _make_execution_kernel(replay)
                a = wp.array(a_host, device=device, requires_grad=True)
                b = wp.array(b_host, device=device, requires_grad=True)
                g = wp.array(g_host, device=device)
                c = wp.empty((32, 32), device=device, requires_grad=True)
                with wp.Tape() as tape:
                    wp.launch_tiled(kernel, dim=1, inputs=[a, b, g], outputs=[c], block_dim=64, device=device)
                tape.backward(grads={c: g})
                np.testing.assert_allclose(c.numpy(), a_host @ b_host, rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(a.grad.numpy(), g_host @ b_host.T, rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(b.grad.numpy(), a_host.T @ g_host, rtol=1e-4, atol=1e-4)

                module = kernel.module.load(device, 64)
                hooks = module.get_kernel_hooks(kernel)
                self.assertTrue(custom.adj.gemm_pipeline_requirements)
                self.assertGreaterEqual(hooks.backward_smem_bytes, custom.adj.get_total_required_shared())
                self.assertEqual(len(module.gemm_pipeline_resources), 2 if replay else 3)

                # Replaying a captured backward call must retain the same hidden
                # resources and the conservative deferred-helper workspace.
                tape.zero()
                g.assign(g_host)
                with wp.ScopedCapture(device, force_module_load=False) as capture:
                    tape.backward(grads={c: g})
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(a.grad.numpy(), g_host @ b_host.T, rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(b.grad.numpy(), a_host.T @ g_host, rtol=1e-4, atol=1e-4)

    def _pipeline_runtime(self):
        device = SimpleNamespace(is_cuda=True, context=None)
        ready = SimpleNamespace(is_complete=False)
        first = SimpleNamespace(cuda_stream=1, record_event=mock.Mock(return_value=ready), wait_event=mock.Mock())
        second = SimpleNamespace(cuda_stream=2, record_event=mock.Mock(), wait_event=mock.Mock())
        module = context.ModuleExec(None, b"test", device, {}, 64)
        return module, first, second, [_Array(device, 16), _Array(device, 32)]

    def test_pipeline_initialization_orders_other_streams(self):
        module, first, second, arrays = self._pipeline_runtime()
        order = []
        ready = first.record_event.return_value
        first.record_event.side_effect = lambda: order.append("ready") or ready
        with (
            mock.patch.object(context, "Array", _Array),
            mock.patch.object(wp, "ScopedStream", side_effect=lambda stream: contextlib.nullcontext()) as scope,
            mock.patch.object(
                wp, "empty", side_effect=lambda *args, **kwargs: order.append("allocate") or SimpleNamespace(ptr=4096)
            ),
            mock.patch.object(
                module, "_launch_gemm_pipeline_helper", side_effect=lambda *args: order.append("initialize")
            ),
            mock.patch.object(context.runtime.core, "wp_cuda_stream_is_capturing", return_value=False) as capturing,
        ):
            first_params = module.get_gemm_pipeline_params([_pipeline_metadata()], arrays, first)
            self.assertEqual(order, ["allocate", "initialize", "ready"])
            scope.assert_called_once_with(first)
            capturing.return_value = True
            second_params = module.get_gemm_pipeline_params([_pipeline_metadata()], arrays, second)
            self.assertEqual(first_params[0].value, second_params[0].value)
            second.wait_event.assert_called_once_with(ready, external=True)
            first.wait_event.assert_not_called()
            self.assertEqual(len(module.gemm_pipeline_resources), 1)
            # Observe completion outside capture, then retain it without adding
            # event nodes to future warmed cross-stream graph captures.
            capturing.return_value = False
            ready.is_complete = True
            module.get_gemm_pipeline_params([_pipeline_metadata()], arrays, second)
            capturing.return_value = True
            module.get_gemm_pipeline_params([_pipeline_metadata()], arrays, second)
            second.wait_event.assert_called_once_with(ready, external=True)

    def test_capture_requires_pipeline_warmup(self):
        module, first, _, arrays = self._pipeline_runtime()
        with (
            mock.patch.object(context, "Array", _Array),
            mock.patch.object(context.runtime.core, "wp_cuda_stream_is_capturing", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                module.get_gemm_pipeline_params([_pipeline_metadata()], arrays, first)
        self.assertFalse(module.gemm_pipeline_resources)

    def test_deterministic_command_rebinds_pipeline(self):
        module, stream, _, arrays = self._pipeline_runtime()
        hooks = context.KernelHooks(
            None,
            None,
            gemm_pipeline_meta=[_pipeline_metadata()],
            forward_smem_bytes=8192,
            gemm_pipeline_smem_bytes=4096,
        )
        args = [SimpleNamespace(type=wp.array2d[float], label=name) for name in ("a", "b")]
        kernel = SimpleNamespace(
            adj=SimpleNamespace(args=args, kernel_dim=1),
            module=SimpleNamespace(load=mock.Mock(return_value=module)),
            grid_stride=False,
        )
        bounds = context.launch_bounds_t((64,))
        params = [bounds, ctypes.c_void_p(16), ctypes.c_void_p(32), ctypes.c_void_p(1234)]
        command = deterministic.create_deterministic_launch(
            kernel,
            module.device,
            SimpleNamespace(scatter_targets=[], counter_targets=[]),
            arrays,
            hooks=hooks,
            params=params,
            bounds=bounds,
            block_dim=64,
        )
        replacement = _Array(module.device, 48)
        with (
            mock.patch.object(context, "Array", _Array),
            mock.patch.object(
                context,
                "pack_arg",
                side_effect=lambda kernel, dtype, name, value, device, adjoint: ctypes.c_void_p(value.ptr),
            ),
            mock.patch.object(wp, "ScopedStream", side_effect=lambda stream: contextlib.nullcontext()),
            mock.patch.object(wp, "empty", return_value=SimpleNamespace(ptr=4096)),
            mock.patch.object(module, "_launch_gemm_pipeline_helper") as initialize,
            mock.patch.object(context.runtime.core, "wp_cuda_stream_is_capturing", return_value=False),
            mock.patch.object(deterministic, "launch_deterministic") as launch,
        ):
            command.set_param_at_index(0, replacement)
            command.launch(stream)
            self.assertEqual(initialize.call_args.args[1][1].value, replacement.ptr)
            self.assertEqual(launch.call_args.args[2][-1].value, 4096)
            self.assertEqual(command._forward_smem_bytes, 4096)


if __name__ == "__main__":
    unittest.main(verbosity=2)
