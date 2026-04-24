// Sample Mosaic IR kernel — vector add on 128-lane f32 VMEM memrefs.
//
// Every memref carries:
//   * shape:         128xf32
//   * layout:        #tpu.tiled<(128),[1]>  (required by Mosaic legalization)
//   * memory space:  #tpu.memory_space<vmem>
//
// Mosaic's legalization pass requires TiledLayoutAttr on every memref
// argument; without it you get "All memref arguments should use the
// TiledLayoutAttr for layout".

module {
  func.func @main(
      %lhs: memref<128xf32, #tpu.tiled<(128),[1]>, #tpu.memory_space<vmem>>,
      %rhs: memref<128xf32, #tpu.tiled<(128),[1]>, #tpu.memory_space<vmem>>,
      %dst: memref<128xf32, #tpu.tiled<(128),[1]>, #tpu.memory_space<vmem>>
  ) {
    %c0 = arith.constant 0 : index
    %v_lhs = vector.load %lhs[%c0]
      : memref<128xf32, #tpu.tiled<(128),[1]>, #tpu.memory_space<vmem>>,
        vector<128xf32>
    %v_rhs = vector.load %rhs[%c0]
      : memref<128xf32, #tpu.tiled<(128),[1]>, #tpu.memory_space<vmem>>,
        vector<128xf32>
    %v_sum = arith.addf %v_lhs, %v_rhs : vector<128xf32>
    vector.store %v_sum, %dst[%c0]
      : memref<128xf32, #tpu.tiled<(128),[1]>, #tpu.memory_space<vmem>>,
        vector<128xf32>
    return
  }
}
