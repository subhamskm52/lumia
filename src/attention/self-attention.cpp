#include "lumia/attention/self-attention.h"
#include <cmath>

namespace lumia::attention {
    SelfAttention::SelfAttention(long dim_size){

        double scale = std::sqrt(6.0 / (dim_size + dim_size)); // Xavier Glorot initialization
        Wq = Eigen::MatrixXd::Random(dim_size, dim_size)*scale;
        Wk = Eigen::MatrixXd::Random(dim_size, dim_size)*scale;
        Wv = Eigen::MatrixXd::Random(dim_size, dim_size)*scale;

    }

    Eigen::MatrixXd SelfAttention::forward(Eigen::MatrixXd x) {
        //  Attention calculation
        //    ├── scores
        //    ├── scaling
        //    ├── softmax
        //    └── context
        Eigen::MatrixXd q = x*Wq;
        Eigen::MatrixXd k = x*Wk;
        Eigen::MatrixXd v = x*Wv;

        Eigen::MatrixXd attention_scores = q*k.transpose();
        auto scale = 1.0 / std::sqrt(Wq.cols());
        attention_scores = attention_scores / scale;
        auto attention_weights = softmax(attention_scores); //normalize using softmax

        //context vector
        auto context_vector = attention_weights*v;
        return context_vector;
    }

}
