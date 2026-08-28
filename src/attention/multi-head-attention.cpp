
#include "lumia/attention/multi-head-attention.h"

namespace lumia::attention {
    MultiHeadAttention::MultiHeadAttention(long dim_size, std::size_t num_heads_)
        :num_heads(num_heads_), head_dim(dim_size/num_heads_)
    {
        double scale = std::sqrt(6.0/(dim_size + dim_size));
        Wq.value = Eigen::MatrixXd::Random(dim_size, dim_size)*scale;
        Wk.value = Eigen::MatrixXd::Random(dim_size, dim_size)*scale;
        Wv.value = Eigen::MatrixXd::Random(dim_size, dim_size)*scale;

        Wq.grad = Eigen::MatrixXd::Zero(dim_size, dim_size);
        Wk.grad = Eigen::MatrixXd::Zero(dim_size, dim_size);
        Wv.grad = Eigen::MatrixXd::Zero(dim_size, dim_size);
    }
    Eigen::MatrixXd MultiHeadAttention::forward(Eigen::MatrixXd x) {
        // calculate  Q,K,V for inp
        input.value = x;
        Q.value = x*Wq.value;
        K.value = x*Wk.value;
        V.value = x*Wv.value;

        Eigen::MatrixXd context_output(Q.value.rows(), Q.value.cols());
        for (int i = 0; i< num_heads; i++) {
            auto start_col = i*head_dim;
            Eigen::MatrixXd Qi = Q.value.block(0, start_col, Q.value.rows(),head_dim);
            Eigen::MatrixXd Ki = K.value.block(0, start_col, K.value.rows(),head_dim);
            Eigen::MatrixXd Vi = V.value.block(0, start_col, V.value.rows(),head_dim);

            // calculate attention scores
            Eigen::MatrixXd attention_scores = Qi*Ki.transpose();

            // scale and normalize using softmax
            double scale = 1.0 / std::sqrt(head_dim);
            auto atten_weight = softmax(attention_scores * scale);
            attention_weights.push_back(atten_weight);

            // context vector
            Eigen::MatrixXd context = atten_weight*Vi;
            context_output.block(0, start_col, Q.value.rows(), head_dim) = context;
        }
        return context_output;
    }

    Eigen::MatrixXd MultiHeadAttention::backward(Eigen::MatrixXd output_grad)
    {
        Eigen::MatrixXd grad_x =
            Eigen::MatrixXd::Zero(
                input.value.rows(),
                input.value.cols()
            );

        double scale =
            1.0 / std::sqrt(head_dim);

        for (std::size_t i = 0; i < num_heads; i++) {

            std::size_t start_col =
                i * head_dim;

            // --------------------------------
            // Get cached Q, K, V
            // --------------------------------

            Eigen::MatrixXd Qi =
                Q.value.block(
                    0,
                    start_col,
                    Q.value.rows(),
                    head_dim
                );

            Eigen::MatrixXd Ki =
                K.value.block(
                    0,
                    start_col,
                    K.value.rows(),
                    head_dim
                );

            Eigen::MatrixXd Vi =
                V.value.block(
                    0,
                    start_col,
                    V.value.rows(),
                    head_dim
                );

            const Eigen::MatrixXd& Ai = attention_weights[i];

            // --------------------------------
            // Gradient for this head
            // --------------------------------

            Eigen::MatrixXd grad_context =
                output_grad.block(
                    0,
                    start_col,
                    output_grad.rows(),
                    head_dim
                );


            // --------------------------------
            // C = A V
            // --------------------------------

            Eigen::MatrixXd grad_A =
                grad_context * Vi.transpose();

            Eigen::MatrixXd grad_V =
                Ai.transpose() * grad_context;


            // --------------------------------
            // A = softmax(S)
            // --------------------------------

            Eigen::MatrixXd grad_S =
                Eigen::MatrixXd::Zero(
                    Ai.rows(),
                    Ai.cols()
                );

            for (int r = 0; r < Ai.rows(); r++) {

                Eigen::VectorXd a =
                    Ai.row(r).transpose();

                Eigen::VectorXd g =
                    grad_A.row(r).transpose();

                double dot =
                    a.dot(g);

                Eigen::VectorXd grad =
                    a.array() *
                    (g.array() - dot);

                grad_S.row(r) =
                    grad.transpose();
            }


            // --------------------------------
            // S = Q Kᵀ / sqrt(head_dim)
            // --------------------------------

            grad_S *= scale;

            Eigen::MatrixXd grad_Q =
                grad_S * Ki;

            Eigen::MatrixXd grad_K =
                grad_S.transpose() * Qi;


            // --------------------------------
            // Q = X Wq
            // --------------------------------

            Wq.grad +=
                input.value.transpose() * grad_Q;

            Eigen::MatrixXd grad_x_Q =
                grad_Q * Wq.value.transpose();


            // --------------------------------
            // K = X Wk
            // --------------------------------

            Wk.grad +=
                input.value.transpose() * grad_K;

            Eigen::MatrixXd grad_x_K =
                grad_K * Wk.value.transpose();


            // --------------------------------
            // V = X Wv
            // --------------------------------

            Wv.grad +=
                input.value.transpose() * grad_V;

            Eigen::MatrixXd grad_x_V =
                grad_V * Wv.value.transpose();


            // --------------------------------
            // Q, K and V all came from X
            // --------------------------------

            grad_x +=
                grad_x_Q +
                grad_x_K +
                grad_x_V;
        }

        return grad_x;
    }

}
