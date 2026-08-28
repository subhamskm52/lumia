#include "lumia/layers/LayerNorm.h"

namespace lumia::layers {
    LayerNorm::LayerNorm(const long size_dim, const float eps_)
        :eps(eps_){
        scale.value = Eigen::VectorXd::Random(size_dim);
        shift.value = Eigen::VectorXd::Random(size_dim);
    }

    Eigen::MatrixXd LayerNorm::forward(Eigen::MatrixXd x) {
        input.value = x;

        // 1. Calculate mean: (seq_len, 1)
        mean = x.rowwise().mean();

        // 2. Center: (seq_len, dim)
        Eigen::MatrixXd centered = x.rowwise() - mean.transpose();

        // 3. Calculate variance: (seq_len, 1)
        variance = centered.array().square().rowwise().mean();

        // 4. Normalize: (seq_len, dim)
        normalized = centered.array().colwise()
                                     / (variance.array() + eps).sqrt();

        // 5. Scale and shift: (seq_len, dim)
        Eigen::MatrixXd output = normalized.array() * scale.value.array();
        output.array() += shift.value.array();

        return output;
    }

    Eigen::MatrixXd LayerNorm::backward(const Eigen::MatrixXd& grad) {

        const int seq_len = input.value.rows();
        const int dim = input.value.cols();

        // --------------------------------
        // dL/dshift
        // --------------------------------
        shift.grad += grad.colwise().sum();

        // --------------------------------
        // dL/dscale
        // --------------------------------
        scale.grad =
            (grad.array() * normalized.array())
            .colwise()
            .sum();

        // --------------------------------
        // dL/dnormalized
        // --------------------------------
        Eigen::MatrixXd grad_norm =
            grad.array() * scale.value.transpose().array();

        // --------------------------------
        // dL/dinput
        // --------------------------------
        Eigen::MatrixXd grad_input(seq_len, dim);

        for (int i = 0; i < seq_len; ++i) {

            const double inv_std =
                1.0 / std::sqrt(variance(i) + eps);

            const double sum_grad =
                grad_norm.row(i).sum();

            const double sum_grad_norm =
                (grad_norm.row(i).array()
                 * normalized.row(i).array()).sum();

            for (int j = 0; j < dim; ++j) {

                grad_input(i, j) =
                    (inv_std / dim) *
                    (
                        dim * grad_norm(i, j)
                        - sum_grad
                        - normalized(i, j) * sum_grad_norm
                    );
            }
        }

        return grad_input;
    }
}
