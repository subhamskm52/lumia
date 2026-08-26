#pragma once
#include <Eigen/Dense>

#include "lumia/model/paramerter.h"

namespace lumia::layers::activation {
    class GELU {
    private:
        static lumia::model::Parameter input;
    public:
        static inline Eigen::MatrixXd forward(const Eigen::MatrixXd& x) {
            input.value = x;

           return x.unaryExpr([](const double val) {
               return 0.5f * val * (1.0f + std::erf(val / std::sqrt(2.0f)));
            });
        }
        static inline Eigen::MatrixXd backward(const Eigen::MatrixXd& grad) {
            return grad.array() *
                   (
                       0.5 * (
                           1.0 +
                           (input.value.array() / std::sqrt(2.0)).erf()
                       )
                       +
                       input.value.array() /
                       std::sqrt(2.0 * M_PI) *
                       (-0.5 * input.value.array().square()).exp()
                   );
        }
    };
}
