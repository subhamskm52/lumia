#include "lumia/layers/transformer/transformer-block.h"

namespace lumia::layers {

TransformerBlock::TransformerBlock(
    std::size_t embedding_dim,
    std::size_t num_heads,
    std::size_t ffn_dim
)
    : norm1(embedding_dim),
      attention(embedding_dim, num_heads),
      norm2(embedding_dim),
      ffn1(embedding_dim, ffn_dim),
      ffn2(ffn_dim, embedding_dim) {
}

Eigen::MatrixXd TransformerBlock::forward(
    const Eigen::MatrixXd& input) {

    // -----------------------------
    // Attention sub-layer
    // -----------------------------

    Eigen::MatrixXd norm1_output =
        norm1.forward(input);

    Eigen::MatrixXd attention_output =
        attention.forward(norm1_output);

    // Residual connection
    Eigen::MatrixXd residual1 =
        input + attention_output;


    // -----------------------------
    // Feed Forward Network
    // -----------------------------

    Eigen::MatrixXd norm2_output =
        norm2.forward(residual1);

    Eigen::MatrixXd ffn_output =
        ffn1.forward(norm2_output);

    ffn_output =
        gelu.forward(ffn_output);

    ffn_output =
        ffn2.forward(ffn_output);

    // Residual connection
    Eigen::MatrixXd output =
        residual1 + ffn_output;

    return output;
}


Eigen::MatrixXd TransformerBlock::backward(
    const Eigen::MatrixXd& grad_output) {

    // ==========================================
    // Second residual:
    //
    // output = residual1 + ffn_output
    //
    // gradient goes to BOTH paths
    // ==========================================

    Eigen::MatrixXd grad_residual1 =
        grad_output;

    Eigen::MatrixXd grad_ffn =
        grad_output;


    // ==========================================
    // FFN backward
    //
    // ffn_output = ffn2(
    //                  GELU(
    //                      ffn1(norm2(residual1))
    //                  )
    //              )
    // ==========================================

    Eigen::MatrixXd grad_ffn2 =
        ffn2.backward(grad_ffn);

    Eigen::MatrixXd grad_gelu =
        gelu.backward(grad_ffn2);

    Eigen::MatrixXd grad_ffn1 =
        ffn1.backward(grad_gelu);

    Eigen::MatrixXd grad_norm2 =
        norm2.backward(grad_ffn1);


    // Gradient through second residual
    grad_residual1 += grad_norm2;


    // ==========================================
    // First residual:
    //
    // residual1 = input + attention_output
    //
    // gradient goes to BOTH paths
    // ==========================================

    Eigen::MatrixXd grad_input =
        grad_residual1;

    Eigen::MatrixXd grad_attention =
        grad_residual1;


    // ==========================================
    // Attention backward
    // ==========================================

    Eigen::MatrixXd grad_norm1 =
        attention.backward(grad_attention);

    Eigen::MatrixXd grad_norm1_input =
        norm1.backward(grad_norm1);


    // Gradient through first residual
    grad_input += grad_norm1_input;


    return grad_input;
}

}