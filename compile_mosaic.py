#!/usr/bin/env python3
"""Compile a Mosaic MLIR kernel into a wrapped StableHLO module.

Usage:
    python compile_mosaic.py vector_add.mlir vector_add_wrapped.mlir

Reads a Mosaic MLIR file (e.g. the vector_add kernel with tpu.* /
vector.* / memref.* ops and VMEM memory spaces), runs it through
jax's `as_tpu_kernel` machinery to serialize it and wrap it in a
StableHLO custom_call op, and writes the resulting StableHLO module
to disk.

The output file is what the C++ program (main.cpp) hands to PJRT's
Compile API. PJRT accepts textual StableHLO; the embedded Mosaic
bytecode rides along as the custom_call's backend_config.

Where this has to run
---------------------

In principle this only serializes IR and doesn't need a TPU. In
practice `as_tpu_kernel` validates the module by invoking Mosaic's
serde pass, which requires jaxlib's TPU dialect integration to be
importable. Running on a TPU VM is the safest bet.

    pip install jax[tpu]  # on a TPU VM

Notes
-----

jaxlib's internal API evolves between versions. The exact call to
extract the wrapped StableHLO has changed in the past. If this
script breaks, the fallback is:

    @jax.jit
    def wrapper(x, y):
        # call the mosaic kernel via tpu_custom_call / as_tpu_kernel
        return ...

    lowered = wrapper.lower(dummy_x, dummy_y)
    wrapped_hlo = lowered.as_text()  # textual StableHLO
    Path("out.mlir").write_text(wrapped_hlo)

which uses the public JAX lowering API and avoids any internal
jaxlib poking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def compile_mosaic(mosaic_mlir_path: Path, output_path: Path) -> None:
    try:
        import jax
        import jax.numpy as jnp
        from jax._src.tpu_custom_call import as_tpu_kernel
        from jaxlib.mlir import ir
        from jaxlib.mosaic.python import tpu as tpu_dialect_py
        from jaxlib.mlir._mlir_libs import _jax_mlir_ext
    except ImportError as e:
        print(f"ERROR: Required JAX/jaxlib imports failed: {e}", file=sys.stderr)
        print("Install with: pip install jax[tpu]", file=sys.stderr)
        sys.exit(1)

    # Set up an MLIR context with the upstream dialects and the TPU
    # dialect registered.
    ctx = ir.Context()
    registry = ir.DialectRegistry()
    _jax_mlir_ext.register_dialects(registry)
    ctx.append_dialect_registry(registry)
    ctx.load_all_available_dialects()
    tpu_dialect_py.register_dialect(ctx)

    # Parse the Mosaic MLIR module.
    module = ir.Module.parseFile(str(mosaic_mlir_path), ctx)
    print(f"Parsed Mosaic module from {mosaic_mlir_path}")

    # Wrap the Mosaic module into a JAX-callable via as_tpu_kernel.
    # We don't actually invoke it here — we just need it to construct
    # the wrapped HLO via JAX's lowering.
    #
    # The output type is inferred from the module's @main signature;
    # for illustration we hard-code a vector<128xf32> output.
    out_type = jax.ShapeDtypeStruct(shape=(128,), dtype=jnp.float32)

    kernel_fn = as_tpu_kernel(
        module,
        out_type=out_type,
        input_output_aliases=(),
        serialization_format=1,
    )

    # Trace the kernel through jax.jit and lower to get the wrapped
    # StableHLO that contains the tpu_custom_call op.
    dummy_lhs = jnp.zeros((128,), dtype=jnp.float32)
    dummy_rhs = jnp.zeros((128,), dtype=jnp.float32)

    @jax.jit
    def wrapper(lhs, rhs):
        return kernel_fn(lhs, rhs)

    lowered = wrapper.lower(dummy_lhs, dummy_rhs)
    wrapped_stablehlo = lowered.as_text()  # textual StableHLO

    output_path.write_text(wrapped_stablehlo)
    print(f"Wrote wrapped StableHLO to {output_path}")
    print(f"  Size: {len(wrapped_stablehlo)} bytes")
    print(f"  First 500 chars:\n{wrapped_stablehlo[:500]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mosaic_mlir",
        type=Path,
        help="Input Mosaic MLIR file (e.g. vector_add.mlir)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output wrapped StableHLO file (e.g. vector_add_wrapped.mlir)",
    )
    args = parser.parse_args()
    compile_mosaic(args.mosaic_mlir, args.output)


if __name__ == "__main__":
    main()
