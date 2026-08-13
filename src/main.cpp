#include <iostream>
#include <tokenier/bpe_tokenizer.h>
#include <tokenier/tokenizer.h>
int main() {

    std::unique_ptr<lumia::BpeTokenizer> tk = std::make_unique<lumia::BpeTokenizer>();

    lumia::Corpus corpus("../data.txt");

    tk->train(corpus, 15);

    // Encode
    std::string text = "lamp";

    auto tokens = tk->encode(text);

    std::cout << "Input: " << text << "\n";

    std::cout << "Tokens: ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << "\n";

    // Decode
    std::string decoded = tk->decode(tokens);

    std::cout << "Decoded: " << decoded << "\n";

}