#include <iostream>
#include <tokenier/bpe_tokenizer.h>
#include <tokenier/tokenizer.h>
int main() {

    std::unique_ptr<lumia::Tokenizer> tk = std::make_unique<lumia::BpeTokenizer>();

    lumia::Corpus corpus("../data.txt");

    tk->train(corpus, 15);

}