#include "lumia/attention/self-attention.h"
#include <cmath>

namespace lumia::attention {
    SelfAttention::SelfAttention(long input_size,long dim_size){

        double scale = std::sqrt(6.0 / (input_size + dim_size)); // Xavier Glorot initialization
        Wq = Eigen::MatrixXd::Random(input_size, dim_size)*scale;
        Wk = Eigen::MatrixXd::Random(input_size, dim_size)*scale;
        Wv = Eigen::MatrixXd::Random(input_size, dim_size)*scale;

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
        attention_scores = attention_scores / std::sqrt(static_cast<double>(Wq.cols()));
        auto attention_weights = softmax(attention_scores); //normalize using softmax

        //context vector
        auto context_vector = attention_weights*v;
        return context_vector;
    }

}
