#pragma once
#include <vector>

namespace lumia {
    class Tokenizer {
    public:
        virtual ~Tokenizer() = default;
        virtual std::vector<int> encode(std::string& text) = 0;
        virtual std::string decode(std::vector<int>& token ) = 0;
    };
}
