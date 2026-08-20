#pragma once
#include <Eigen/Dense>

namespace lumia::layers {

    class LayerNorm {
    public:
        LayerNorm(long size_dim, float eps_ = 1e-5f);
        Eigen::MatrixXd forward(Eigen::MatrixXd x);
    private:
        Eigen::VectorXd scale;
        Eigen::VectorXd shift;
        float eps;
    };

}