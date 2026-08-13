#include "tokenier/bpe_tokenizer.h"

#include <iostream>

namespace lumia {
    void BpeTokenizer::train(const lumia::Corpus& word_corpus_,std::size_t vocab_size){

        std::vector<std::vector<std::string>> corpus;
        for (auto& words : word_corpus_.data()) {
            std::vector<std::string> chr_corpus;
            chr_corpus.reserve(words.size());

            for (auto& word: words) {
                for (char chr: word) {
                    std::string chr_str = std::string(1, chr);
                    chr_corpus.emplace_back(chr_str);
                    if (!vocab.contains(chr_str)) {
                        vocab[chr_str] = vocab.size()+1;
                    }
                }
            }
            corpus.emplace_back(chr_corpus);
        }

        while (vocab.size() < vocab_size) {

            std::map<std::string, size_t> pairs;

            // Count pairs
            for (const auto& words : corpus) {
                for (size_t i = 0; i + 1 < words.size(); ++i) {
                    std::string pair = words[i] + words[i + 1];
                    ++pairs[pair];
                }
            }

            // Nothing left to merge
            if (pairs.empty()) {
                break;
            }

            // Find most frequent pair
            const auto max_it = std::max_element(
                pairs.begin(),
                pairs.end(),
                [](const auto& a, const auto& b) {
                    return a.second < b.second;
                }
            );

            const std::string best_pair = max_it->first;

            // Add new token
            if (!vocab.contains(best_pair)) {
                vocab[best_pair] = vocab.size()+1;
            }

            // Merge pair everywhere
            for (auto& word : corpus) {
                for (size_t i = 0; i + 1 < word.size();) {

                    if (word[i] + word[i + 1] == best_pair) {
                        word[i] = best_pair;
                        word.erase(word.begin() + i + 1);
                    } else {
                        ++i;
                    }
                }
            }
        }

    }

    std::string BpeTokenizer::decode(std::vector<int> &token) {
        return "";
    }

    std::vector<int> BpeTokenizer::encode(std::string &text) {
        return {};
    }
}
