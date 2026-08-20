
#include "lumia/attention/multi-head-attention.h"

namespace lumia::attention {
    MultiHeadAttention::MultiHeadAttention(long dim_size, std::size_t num_heads_)
        :num_heads(num_heads_), head_dim(dim_size/num_heads_)
    {
        double scale = std::sqrt(6.0/(dim_size + dim_size));
        Wq = Eigen::MatrixXd::Random(dim_size, dim_size)*scale;
        Wk = Eigen::MatrixXd::Random(dim_size, dim_size)*scale;
        Wv = Eigen::MatrixXd::Random(dim_size, dim_size)*scale;
    }
    Eigen::MatrixXd MultiHeadAttention::forward(Eigen::MatrixXd x) {
        // calculate  Q,K,V for inp
        auto Q = x*Wq;
        auto K = x*Wk;
        auto V = x*Wv;

        Eigen::MatrixXd context_output(Q.rows(), Q.cols());
        for (int i = 0; i< num_heads; i++) {
            auto start_col = i*head_dim;
            Eigen::MatrixXd Qi = Q.block(0, start_col, Q.rows(),head_dim);
            Eigen::MatrixXd Ki = K.block(0, start_col, K.rows(),head_dim);
            Eigen::MatrixXd Vi = V.block(0, start_col, V.rows(),head_dim);

            // calculate attention scores
            Eigen::MatrixXd attention_scores = Qi*Ki.transpose();

            // scale and normalize using softmax
            double scale = 1.0 / std::sqrt(head_dim);
            Eigen::MatrixXd attention_weights = softmax(attention_scores * scale);

            // context vector
            Eigen::MatrixXd context = attention_weights*Vi;
            context_output.block(0, start_col, Q.rows(), head_dim) = context;
        }
        return context_output;
    }

}
