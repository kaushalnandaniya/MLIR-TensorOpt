// Sample Attention Head block in Linalg-on-Tensors Dialect
// RUN: mlir-tensoropt-opt %s --mlir-tensoropt-attention-fusion --mlir-tensoropt-loop-tiling="tile-sizes=64,64,32" --mlir-tensoropt-vectorize="vector-width=128"

module {
  func.func @attention_head_fwd(
      %Q: tensor<64x32xf32>,
      %K: tensor<32x64xf32>,
      %V: tensor<64x32xf32>,
      %Out: tensor<64x32xf32>) -> tensor<64x32xf32> {
    
    // 1. Q @ K^T Attention Logits
    %init_logits = tensor.empty() : tensor<64x64xf32>
    %logits = linalg.matmul
      ins(%Q, %K : tensor<64x32xf32>, tensor<32x64xf32>)
      outs(%init_logits : tensor<64x64xf32>) -> tensor<64x64xf32>

    // 2. Softmax weights @ V
    %res = linalg.matmul
      ins(%logits, %V : tensor<64x64xf32>, tensor<64x32xf32>)
      outs(%Out : tensor<64x32xf32>) -> tensor<64x32xf32>

    return %res : tensor<64x32xf32>
  }
}
