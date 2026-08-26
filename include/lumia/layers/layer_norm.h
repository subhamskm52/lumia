#pragma once
#include <Eigen/Dense>

#include "lumia/model/paramerter.h"

namespace lumia::layers {

    class layer_norm {
    public:
        layer_norm(long size_dim, float eps_ = 1e-5f);
        Eigen::MatrixXd forward(Eigen::MatrixXd x);
        Eigen::MatrixXd backward(const Eigen::MatrixXd& grad);

    private:
        float eps;

        Eigen::VectorXd mean;
        Eigen::VectorXd variance;
        Eigen::MatrixXd normalized;

        lumia::model::Parameter input;
        lumia::model::Parameter scale;
        lumia::model::Parameter shift;
    };

}