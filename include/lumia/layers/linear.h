#pragma once
#include <Eigen/Dense>
namespace lumia::layers {
    class Linear {
    public:
        Linear(std::size_t in_features, std::size_t out_features);
        Eigen::MatrixXd forward(const Eigen::MatrixXd& x);
    private:
        Eigen::MatrixXd weights;
        Eigen::MatrixXd bias;
    };
}