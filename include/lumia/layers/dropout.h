#pragma once
#include <Eigen/Dense>
#include <random>
#include <stdexcept>

namespace lumia::layers {
    class Dropout {
    public:
        explicit Dropout(const double drop_rate):drop_rate_(drop_rate) {
            if (drop_rate < 0.0f || drop_rate >= 1.0f) {
                throw std::invalid_argument(
                    "drop_rate must be in [0, 1)"
                );
            }
        };


        Eigen::MatrixXd forward(
            const Eigen::MatrixXd& x) {

            Eigen::MatrixXd output = x;

            static std::mt19937 generator{
                std::random_device{}()
            };

            std::bernoulli_distribution keep(
                1.0 - drop_rate_
            );

            const float scale = 1.0f / (1.0f - drop_rate_);

            for (Eigen::Index i = 0; i < output.rows(); ++i) {
                for (Eigen::Index j = 0; j < output.cols(); ++j) {

                    if (keep(generator)) {
                        output(i, j) *= scale;
                    } else {
                        output(i, j) = 0.0f;
                    }
                }
            }
            return output;
        }

    private:
        double drop_rate_;
    };

}