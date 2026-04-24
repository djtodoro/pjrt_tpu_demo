// Sample Mosaic IR kernel — vector add on 128-lane f32 VMEM memrefs.
//
// Conventions (libtpu 0.23 / Aug 2023):
//
//   * Function args carry a TiledLayoutAttr:
//         memref<128xf32, #tpu.tiled<RANK,(tile_dims...)>, #tpu.memory_space<vmem>>
//     Required by Mosaic's "All memref arguments should use the TiledLayoutAttr".
//
//   * Inner uses (e.g. vector.load) see a layout-erased memref. Use
//         tpu.erase_memref_layout %arg : <typed> -> <plain>
//     to convert from the tiled-layout memref to the plain one. Required
//     because upstream MLIR's vector.load verifier checks for unit stride
//     on the innermost dim, which a non-linear tiled layout breaks.
//
//   * For rank-1 f32 memrefs of 128 elements on TPU v4, the canonical tile
//     is (128) and the layout is `#tpu.tiled<1,(128)>`.

module {
  func.func @main(
      %lhs_tiled: memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>,
      %rhs_tiled: memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>,
      %dst_tiled: memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>
  ) {
    %lhs = tpu.erase_memref_layout %lhs_tiled
        : memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>
       -> memref<128xf32, #tpu.memory_space<vmem>>
    %rhs = tpu.erase_memref_layout %rhs_tiled
        : memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>
       -> memref<128xf32, #tpu.memory_space<vmem>>
    %dst = tpu.erase_memref_layout %dst_tiled
        : memref<128xf32, #tpu.tiled<1,(128)>, #tpu.memory_space<vmem>>
       -> memref<128xf32, #tpu.memory_space<vmem>>

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
