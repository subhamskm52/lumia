#include "tokenier/bpe_tokenizer.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace lumia {
    void BpeTokenizer::train(const lumia::Corpus& word_corpus_,std::size_t vocab_size){
        vocab[" "]=1;
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


    void BpeTokenizer::load(const std::string& path) {

        std::ifstream file(path);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }
        vocab.clear();
        std::string line;
        while (std::getline(file, line)) {

            if (line.empty()) {
                continue;
            }

            std::stringstream ss(line);

            std::string token;
            std::size_t id;

            ss >> token >> id;

            vocab[token] = id;
        }
    }

    void BpeTokenizer::save(const std::string& path) const {
        std::ofstream file(path);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        for (const auto& [token, id] : vocab) {
            file << token << '\t' << id << '\n';
        }
    }
    std::vector<int> BpeTokenizer::encode(std::string& text) {

        std::vector<std::string> tokens;

        // Start with character tokens
        for (char c : text) {
            tokens.emplace_back(1, c);
        }

        // Sliding window of 2
        for (int i = 0; i + 1 < static_cast<int>(tokens.size()); ++i) {

            std::string pair = tokens[i] + tokens[i + 1];

            if (vocab.contains(pair)) {
                // Merge
                tokens[i] = pair;
                tokens.erase(tokens.begin() + i + 1);
                i--;
            }
        }

        // Convert tokens -> IDs
        std::vector<int> result;
        result.reserve(tokens.size());

        for (const auto& token : tokens) {
            auto it = vocab.find(token);
            if (it == vocab.end()) {
                throw std::runtime_error(
                    "Token not found in vocabulary: [" + token + "]"
                );
            }
            result.push_back(
                static_cast<int>(it->second)
            );
        }

        return result;
    }

    std::string BpeTokenizer::decode(std::vector<int>& tokens) {

        std::map<std::size_t, std::string> reverse_vocab;

        // Create ID -> token mapping
        for (const auto& [token, id] : vocab) {
            reverse_vocab[id] = token;
        }

        std::string result;

        for (int id : tokens) {

            auto it = reverse_vocab.find(static_cast<std::size_t>(id));

            if (it == reverse_vocab.end()) {
                throw std::runtime_error(
                    "Unknown token id: " + std::to_string(id)
                );
            }

            result += it->second;
        }

        return result;
    }
}
