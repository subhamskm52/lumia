#include "lumia/layers/layer_norm.h"

namespace lumia::layers {
    layer_norm::layer_norm(const long size_dim, const float eps_)
        :eps(eps_){
        scale.value = Eigen::VectorXd::Random(size_dim);
        shift.value = Eigen::VectorXd::Random(size_dim);
    }

    Eigen::MatrixXd layer_norm::forward(Eigen::MatrixXd x) {
        // 1. Calculate mean: (seq_len, 1)
        Eigen::VectorXd mean = x.rowwise().mean();

        // 2. Center: (seq_len, dim)
        Eigen::MatrixXd centered = x.rowwise() - mean.transpose();

        // 3. Calculate variance: (seq_len, 1)
        Eigen::VectorXd variance = centered.array().square().rowwise().mean();

        // 4. Normalize: (seq_len, dim)
        Eigen::MatrixXd normalized = centered.array().colwise()
                                     / (variance.array() + eps).sqrt();

        // 5. Scale and shift: (seq_len, dim)
        Eigen::MatrixXd output = normalized.array() * scale.value.array();
        output.array() += shift.value.array();

        return output;
    }
}
