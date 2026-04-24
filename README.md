# pjrt-demo

Takes a Mosaic MLIR kernel, wraps it in a StableHLO `custom_call`, hands
it to PJRT, allocates buffers, executes on TPU, and reads results back.

## Build

Requires a local XLA checkout for the PJRT C API header.

```
cmake -B build -DXLA_SRC_PATH=/path/to/xla
cmake --build build
```

The binary lands at `build/pjrt-demo`.

## Run

On a TPU VM (tested with `pip install torch-xla[tpu]`):

```
./build/pjrt-demo <mosaic_mlir_file> <libtpu_so> [<output_blob_path>]
```

Example (with the sample kernel and a typical `libtpu.so` location):

```
./build/pjrt-demo vector_add.mlir \
    /home/$USER/vm/lib/python3.10/site-packages/libtpu/libtpu.so
```

## Dumping the compiled blob

If a third argument is given, the compiled PJRT executable is serialized
to that path after compilation:

```
./build/pjrt-demo vector_add.mlir /path/to/libtpu.so vector_add.pjrtexec
```

The result is a byte-for-byte serialization of the post-libtpu compile
artifact — useful for caching, shipping, or deserializing with
`PJRT_Executable_DeserializeAndLoad` in a follow-up run. Note the
serialization is **platform-specific and not stable across libtpu
versions** (per the PJRT C API docs).

## Locating libtpu.so

```
find / -name "libtpu*.so*" 2>/dev/null
```

Pick the one where `nm -D <path> | grep GetPjrtApi` returns a hit. On a
`pip install jax[tpu]` or `torch-xla[tpu]` VM it's typically at:

```
<venv>/lib/python3.X/site-packages/libtpu/libtpu.so
```

## Expected output

For `vector_add.mlir` (128-element f32 add, `out[i] = i + 1`):

```
[9] Downloaded output

Result (first 8 elements):
  1 2 3 4 5 6 7 8
Expected:
  1 2 3 4 5 6 7 8

[done]
```
