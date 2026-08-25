#pragma once
#include <Eigen/Dense>

namespace lumia::model {
    class Parameter {
    public:
        Eigen::MatrixXd value;
        Eigen::MatrixXd grad;
    };
}