#include "lumia/layers/linear.h"

namespace lumia::layers {
    Linear::Linear(std::size_t in_features, std::size_t out_features) {
        weights.value = Eigen::MatrixXd::Random(out_features, in_features);
        bias.value = Eigen::MatrixXd::Random(out_features,1);
    }

    Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd& x) {
        input.value = x;
        return x*weights.value.transpose() + bias.value.transpose();
    }

    Eigen::MatrixXd Linear::backward(const Eigen::MatrixXd &grad) {
        // Y = XW + b
        // grad = dL/dY

        // dL/dW = X^T * grad
        // dL/db = sum(grad)
        weights.grad = input.value.transpose() * grad;
        bias.grad = grad.colwise().sum();

        // dL/dX = grad * W^T
        input.grad =  grad * weights.value.transpose();
        return input.grad;
    }
}
