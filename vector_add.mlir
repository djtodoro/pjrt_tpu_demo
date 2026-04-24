// Sample Mosaic IR kernel for the compile_mosaic.py script.
//
// Two 128-lane float32 VMEM inputs, one 128-lane VMEM output.
// Output = lhs + rhs.
//
// Matches the reference Mosaic MLIR dump captured from JAX's Pallas
// for the same computation (see docs/google-tpu-working-notes.md
// section 4.1).

module {
  func.func @main(
      %lhs: memref<128xf32, #tpu.memory_space<vmem>>,
      %rhs: memref<128xf32, #tpu.memory_space<vmem>>,
      %dst: memref<128xf32, #tpu.memory_space<vmem>>
  ) {
    %c0 = arith.constant 0 : index
    %v_lhs = vector.load %lhs[%c0]
      : memref<128xf32, #tpu.memory_space<vmem>>, vector<128xf32>
    %v_rhs = vector.load %rhs[%c0]
      : memref<128xf32, #tpu.memory_space<vmem>>, vector<128xf32>
    %v_sum = arith.addf %v_lhs, %v_rhs : vector<128xf32>
    vector.store %v_sum, %dst[%c0]
      : memref<128xf32, #tpu.memory_space<vmem>>, vector<128xf32>
    return
  }
}
