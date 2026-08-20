#include <iostream>
#include <Eigen/Dense>
#include "lumia/attention/multi-head-attention.h"  // Your header file

int main() {
    // 1. Create the attention module
    int dim = 512;
    int num_heads = 8;
    lumia::attention::MultiHeadAttention mha(dim, num_heads);

    // 2. Create dummy input
    int seq_len = 10;
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(seq_len, dim);

    // 3. Forward pass
    Eigen::MatrixXd output = mha.forward(input);

    // 4. Print results
    std::cout << "Input shape: " << input.rows() << " x " << input.cols() << std::endl;
    std::cout << "Output shape: " << output.rows() << " x " << output.cols() << std::endl;

    std::cout<<output<<std::endl;
    return 0;
}