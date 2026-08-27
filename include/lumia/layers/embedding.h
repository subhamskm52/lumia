#pragma once
#include "lumia/model/paramerter.h"
#include <Eigen/Dense>
namespace lumia::layers {
    class Embedding {
    private:
        std::vector<lumia::model::Parameter> weights;
        Eigen::VectorXd input_tokens;
    public:
        explicit Embedding(std::size_t vocab_size, std::size_t dim_size);
        Eigen::MatrixXd forward(const Eigen::VectorXd& input_tokens);
        void backward(const Eigen::MatrixXd& grad_output);
    };
}
