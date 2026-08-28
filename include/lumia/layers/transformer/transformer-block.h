#pragma once
#pragma once

#include <Eigen/Dense>

#include "lumia/layers/attention/multi_head_attention.h"
#include "lumia/layers/normalization/layer_norm.h"
#include "lumia/layers/linear.h"
#include "lumia/layers/activation/gelu.h"

namespace lumia::t {

    class TransformerBlock {
    private:
        lumia::layers::LayerNorm norm1;
        lumia::layers::MultiHeadAttention attention;

        lumia::layers::LayerNorm norm2;

        lumia::layers::Linear ffn1;
        lumia::layers::GELU gelu;
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