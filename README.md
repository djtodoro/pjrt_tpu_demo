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

### Matching the XLA commit to your libtpu

The PJRT C API evolves. The XLA checkout you build against must match
the PJRT ABI version embedded in your `libtpu.so`, or `PJRT_Client_Create`
will fail with a message like:

```
Unexpected PJRT_Client_Create_Args size: expected 72, got 88.
Check installed software versions. The framework PJRT API version is 0.23.
```

For libtpu from `torch-xla 2.1.0` / `libtpu-nightly 0.1.dev20230825`
(PJRT API 0.23), checkout an XLA commit from that date:

```
cd /path/to/xla
git branch xla-current-main                              # save your current HEAD
git checkout -b xla-v0.23 988ef292c4a4e1eefbd5141f52fefde129138660
```

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
