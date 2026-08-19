#include <iostream>
#include <lumia/tokenier/bpe_tokenizer.h>
#include <lumia/tokenier/tokenizer.h>
int main() {

    std::unique_ptr<lumia::BpeTokenizer> tk = std::make_unique<lumia::BpeTokenizer>();

    lumia::Corpus corpus("/Users/subham/Documents/Head First Design Patterns - Building Extensible and Maintainable Object-Oriented Software - Eric Freeman, Elisabeth Robson - O'Reilly Media (2020).pdf");

    tk->train(corpus, 50000);

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