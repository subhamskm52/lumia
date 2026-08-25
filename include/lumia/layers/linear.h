#pragma once
#include <Eigen/Dense>

#include "lumia/model/paramerter.h"

namespace lumia::layers {
    class Linear {
    public:
        Linear(std::size_t in_features, std::size_t out_features);
        Eigen::MatrixXd forward(const Eigen::MatrixXd& x);
    private:
        lumia::model::Parameter weights;
        lumia::model::Parameter bias;


    };
}
