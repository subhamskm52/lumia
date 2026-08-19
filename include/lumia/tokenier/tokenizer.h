#pragma once
#include <vector>

namespace lumia {
    class Tokenizer {
    public:
        virtual ~Tokenizer() = default;
        virtual std::vector<int> encode(std::string& text) = 0;
        virtual std::string decode(std::vector<int>& token ) = 0;
        virtual void train(const lumia::tokenizer::Corpus& word_corpus_, std::size_t vocab_size) = 0;
    };
}
