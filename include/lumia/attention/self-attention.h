#pragma once
#include "Eigen/Dense"

namespace lumia::attention {
    class SelfAttention {
    public:
        SelfAttention(long dim_size);
        Eigen::MatrixXd forward(Eigen::MatrixXd x);

    private:
        Eigen::MatrixXd Wq;
        Eigen::MatrixXd Wk;
        Eigen::MatrixXd Wv;
        static Eigen::MatrixXd softmax(const Eigen::MatrixXd& x) {
            Eigen::MatrixXd exp_x = x.array().exp(); // Calculate exp for each element
            Eigen::VectorXd row_sums = exp_x.rowwise().sum(); // Sum exp for each row
            return exp_x.array().colwise() / row_sums.array(); // Divide each element by its row sum
        }
    };

}