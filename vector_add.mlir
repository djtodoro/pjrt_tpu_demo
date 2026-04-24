// Sample Mosaic IR kernel — vector add on 128-lane f32 VMEM memrefs.
//
// Every memref carries:
//   * shape:         128xf32
//   * layout:        #tpu.tiled<1,(128)>  (rank,tiles — old libtpu syntax)
//   * memory space:  #tpu.memory_space<vmem>
//
// Mosaic legalization requires TiledLayoutAttr on every memref argument.
// Syntax for libtpu PJRT API 0.23 (Aug 2023):
//     #tpu.tiled<RANK,(tile_dim1,tile_dim2,...)(more_tiles)...>
// The newer XLA syntax (2024+) uses `<(tiles),[strides]>` without rank.

module {
  func.func @main(
      %lhs: memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>,
      %rhs: memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>,
      %dst: memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>
  ) {
    %c0 = arith.constant 0 : index
    %v_lhs = vector.load %lhs[%c0]
      : memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>,
        vector<128xf32>
    %v_rhs = vector.load %rhs[%c0]
      : memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>,
        vector<128xf32>
    %v_sum = arith.addf %v_lhs, %v_rhs : vector<128xf32>
    vector.store %v_sum, %dst[%c0]
      : memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>,
        vector<128xf32>
    return
  }
}
