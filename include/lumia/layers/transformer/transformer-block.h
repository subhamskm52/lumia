#pragma once

#include <Eigen/Dense>

#include "lumia/attention/multi-head-attention.h"
#include "lumia/layers/LayerNorm.h"
#include "lumia/layers/linear.h"
#include "lumia/layers/activations/gelu.h"

namespace lumia::layers {

    class TransformerBlock {
    private:
        layers::LayerNorm norm1;
        lumia::attention::MultiHeadAttention attention;

        lumia::layers::LayerNorm norm2;

        lumia::layers::Linear ffn1;
        lumia::layers::activation::GELU gelu;
        lumia::layers::Linear ffn2;

    public:
        TransformerBlock(
            std::size_t embedding_dim,
            std::size_t num_heads,
            std::size_t ffn_dim
        );

        Eigen::MatrixXd forward(
            const Eigen::MatrixXd& input
        );

        Eigen::MatrixXd backward(
            const Eigen::MatrixXd& grad_output
        );
    };

}