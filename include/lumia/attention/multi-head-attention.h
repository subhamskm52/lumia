#pragma once
#include <Eigen/Dense>
namespace lumia::attention {
    class MultiHeadAttention {
    public:
        MultiHeadAttention(long dim_size, std::size_t num_heads_);
        Eigen::MatrixXd forward(Eigen::MatrixXd x);
    private:
        Eigen::MatrixXd Wq;
        Eigen::MatrixXd Wk;
        Eigen::MatrixXd Wv;
        std::size_t num_heads;
        std::size_t head_dim;
        static Eigen::MatrixXd softmax(const Eigen::MatrixXd& x) {
            Eigen::MatrixXd exp_x = x.array().exp(); // Calculate exp for each element
            Eigen::VectorXd row_sums = exp_x.rowwise().sum(); // Sum exp for each row
            return exp_x.array().colwise() / row_sums.array(); // Divide each element by its row sum
        }
    };
}