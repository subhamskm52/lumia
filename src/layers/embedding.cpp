#include "lumia/layers/embedding.h"

namespace lumia::layers {
    Embedding::Embedding(std::size_t vocab_size, std::size_t embedding_dim) {
        weights.resize(vocab_size);
        for (auto& parameter : weights) {
            parameter.value = Eigen::MatrixXd::Random(1, embedding_dim);
            parameter.grad = Eigen::MatrixXd::Zero(1, embedding_dim);
        }
    }

    Eigen::MatrixXd Embedding::forward(const Eigen::MatrixXd& input_tokens) {

        this->input_tokens = input_tokens;
        const std::size_t sequence_length = input_tokens.rows();
        const std::size_t embedding_dim = weights[0].value.cols();

        Eigen::MatrixXd output(sequence_length, embedding_dim);

        for (std::size_t i = 0; i < sequence_length; ++i) {

            const auto token_id =
                static_cast<std::size_t>(input_tokens(i, 0));

            if (token_id >= weights.size()) {
                throw std::out_of_range("Token ID exceeds vocabulary size");
            }

            output.row(i) = weights[token_id].value;
        }

        return output;
    }

    void Embedding::backward(const Eigen::MatrixXd& grad_output) {

        for (std::size_t i = 0; i < grad_output.rows(); ++i) {
            const auto token_id =
                static_cast<std::size_t>(input_tokens(i, 0));
            weights[token_id].grad += grad_output.row(i);
        }
    }

}
