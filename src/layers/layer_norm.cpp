#include "lumia/layers/layer_norm.h"

namespace lumia::layers {
    layer_norm::layer_norm(const long size_dim, const float eps_)
        :eps(eps_){
        scale = Eigen::VectorXd::Random(size_dim);
        shift = Eigen::VectorXd::Random(size_dim);
    }

    Eigen::MatrixXd layer_norm::forward(Eigen::MatrixXd x) {
        // calculate mean
        Eigen::VectorXd mean = x.rowwise().mean();
        // calculate centered values around 0 i.e (x - mean)
        Eigen::MatrixXd centered = (x.colwise() - mean);

        // calculate variance
        Eigen::VectorXd variance = centered.array().square().rowwise().mean();

        // normalized = (x - mean) / √(variance + ε)
        Eigen::MatrixXd normalized =
            centered.array().colwise()
            / (variance.array() + eps).sqrt();

        // scale and shift
        Eigen::MatrixXd output =
            normalized.array().rowwise() * scale.transpose().array();
        output.rowwise() += shift.transpose();

        return output;
    }
}
