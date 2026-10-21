// Sample MatMul in Linalg-on-Tensors Dialect
module {
  func.func @matmul_f32(%A: tensor<128x64xf32>, %B: tensor<64x256xf32>, %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %res = linalg.matmul
      ins(%A, %B : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%C : tensor<128x256xf32>) -> tensor<128x256xf32>
    return %res : tensor<128x256xf32>
  }
}
