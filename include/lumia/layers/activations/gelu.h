#pragma once
#include <Eigen/Dense>

namespace lumia::layers::activation {
    class GELU {
    public:
        static inline Eigen::MatrixXd forward(const Eigen::MatrixXd& x) {

           return x.unaryExpr([](const double val) {
               return 0.5f * val * (1.0f + std::erf(val / std::sqrt(2.0f)));
            });
        }
    };
}