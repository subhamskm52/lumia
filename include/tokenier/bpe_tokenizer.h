#pragma once
#include <map>

#include "corpus.h"
#include "tokenizer.h"

namespace lumia {

    class BpeTokenizer : public Tokenizer {
    public:
        void train(const lumia::Corpus& c, std::size_t vocab_size) override;
        std::vector<int> encode(std::string& text) override;
        std::string decode(std::vector<int>& token) override;
    private:
        std::map<std::string, std::size_t> vocab;
    };
}
