#include <iostream>
#include "lumia/attention/self-attention.h"
int main() {

    lumia::attention::SelfAttention sf(4,4);
    sf.print_weights();
}