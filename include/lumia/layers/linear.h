#pragma once
#include <Eigen/Dense>

#include "lumia/model/paramerter.h"

namespace lumia::layers {
    class Linear {
    public:
        Linear(std::size_t in_features, std::size_t out_features);
        Eigen::MatrixXd forward(const Eigen::MatrixXd& x);
        Eigen::MatrixXd backward(const Eigen::MatrixXd& grad);
    private:
        lumia::model::Parameter input;
        lumia::model::Parameter weights;
        lumia::model::Parameter bias;
    };
}
