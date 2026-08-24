#pragma once
#include <Eigen/Dense>

namespace lumia::utils {
    class Softmax {
    public:
        static Eigen::MatrixXd calculate(
    const Eigen::MatrixXd& x)
        {
            // Maximum value of each row
            Eigen::VectorXd row_max =
                x.rowwise().maxCoeff();

            // Subtract row maximum
            Eigen::MatrixXd shifted =
                x.colwise() - row_max;

            // exp(x - max)
            Eigen::MatrixXd exp_x =
                shifted.array().exp();

            // Sum each row
            Eigen::VectorXd row_sums =
                exp_x.rowwise().sum();

            // Normalize each row
            return exp_x.array().colwise()
                / row_sums.array();
        }
    };
}