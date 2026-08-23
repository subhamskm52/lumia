#include "lumia/layers/linear.h"

namespace lumia::layers {
    Linear::Linear(std::size_t in_features, std::size_t out_features) {
        weights = Eigen::MatrixXd::Random(out_features, in_features);
        bias = Eigen::MatrixXd::Random(out_features,1);
    }

    Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd& x) {
        return x*weights.transpose() + bias.transpose();
    }
}
