#include "lumia/tokenier/corpus.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lumia::tokenizer {

    Corpus::Corpus(const std::string& path) {
        std::ifstream file(path);

        if (!file.is_open()) throw std::runtime_error("Failed to open corpus file: " + path);

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream stream(line);
            std::vector<std::string> tokens;
            std::string token;

            while (stream >> token) {
                tokens.emplace_back(token);
            }
            corpus_.emplace_back(tokens);
        }
    }
    const std::vector<std::vector<std::string>>&
    Corpus::data() const {
        return corpus_;
    }
}