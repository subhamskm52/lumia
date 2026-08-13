#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "tokenier/bpe_tokenizer.h"
#include "tokenier/corpus.h"

int main() {

    lumia::BpeTokenizer tokenizer;

    lumia::Corpus corpus("../data.txt");

    tokenizer.train(corpus, 5000);

    std::vector<std::string> tests = {

        // Small cases
        "lamp",
        "hello",
        "world",
        "lower",
        "computer",
        "tokenizer",

        // Multiple words
        "hello world",
        "machine learning",
        "large language model",

        // Longer sentences
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is changing the way we build intelligent software.",
        "A tokenizer converts human readable text into a sequence of tokens.",

        // Multiple sentences
        "The model receives a sequence of tokens. It then predicts the next token.",
        "Machine learning requires data and computation. Large language models can learn complex patterns from text.",
        "Byte pair encoding starts with small tokens. Frequent pairs are repeatedly merged into larger tokens."
    };

    for (std::string &original : tests) {

        auto encoded = tokenizer.encode(original);
        auto decoded = tokenizer.decode(encoded);

        // Main round-trip test
        assert(decoded == original);

        std::cout << "[PASS] "
                  << original
                  << "\n";

        std::cout << "  Tokens: ";

        for (auto id : encoded) {
            std::cout << id << ' ';
        }

        std::cout << "\n";
        std::cout << "  Decoded: " << decoded << "\n\n";
    }

    std::cout << "All tokenizer tests passed!\n";

    return 0;
}