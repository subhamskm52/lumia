#pragma once
#include <string>
#include <vector>

namespace lumia {
    class Corpus {
    public:
        explicit Corpus(const std::string& path);
        const std::vector<std::vector<std::string>>& data() const;

    private:
        std::vector<std::vector<std::string>> corpus_;
    };
}
