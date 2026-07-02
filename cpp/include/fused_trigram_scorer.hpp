#pragma once

#include <cstdint>
#include <string>
#include <vector>

class FusedTrigramScorer
{
public:
    explicit FusedTrigramScorer(const std::string &model_path);

    double score(const std::string &url) const;
    bool predict(const std::string &url, double threshold = 0.5) const;

private:
    std::uint64_t n_features_;
    double intercept_;
    std::vector<double> weights_;

    static std::string normalize_url(const std::string &url);
    static std::uint64_t trigram_hash(const std::string &text, std::size_t start);
    static std::uint64_t rolling_update(std::uint64_t prev_hash, unsigned char left_char, unsigned char right_char);
};
