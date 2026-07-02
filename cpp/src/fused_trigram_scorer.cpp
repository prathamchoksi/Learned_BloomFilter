#include "fused_trigram_scorer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace
{
    constexpr std::uint64_t ROLLING_BASE = 1315423911ULL;
    constexpr std::uint64_t UINT64_MASK = 0xFFFFFFFFFFFFFFFFULL;
}

FusedTrigramScorer::FusedTrigramScorer(const std::string &model_path)
    : n_features_(0), intercept_(0.0)
{
    std::ifstream input(model_path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("Failed to open fused model file: " + model_path);
    }

    input.read(reinterpret_cast<char *>(&n_features_), sizeof(n_features_));
    input.read(reinterpret_cast<char *>(&intercept_), sizeof(intercept_));
    if (!input)
    {
        throw std::runtime_error("Failed to read fused model header: " + model_path);
    }

    weights_.resize(static_cast<std::size_t>(n_features_));
    input.read(reinterpret_cast<char *>(weights_.data()), static_cast<std::streamsize>(weights_.size() * sizeof(double)));
    if (!input)
    {
        throw std::runtime_error("Failed to read fused model weights: " + model_path);
    }
}

double FusedTrigramScorer::score(const std::string &url) const
{
    const std::string normalized = normalize_url(url);
    if (normalized.size() < 3 || n_features_ == 0)
    {
        return intercept_;
    }

    double logit = intercept_;
    std::uint64_t trigram = trigram_hash(normalized, 0);
    logit += weights_[static_cast<std::size_t>(trigram % n_features_)];

    for (std::size_t i = 3; i < normalized.size(); ++i)
    {
        trigram = rolling_update(trigram, static_cast<unsigned char>(normalized[i - 3]), static_cast<unsigned char>(normalized[i]));
        logit += weights_[static_cast<std::size_t>(trigram % n_features_)];
    }

    return logit;
}

bool FusedTrigramScorer::predict(const std::string &url, double threshold) const
{
    const double logit = score(url);
    const double probability = 1.0 / (1.0 + std::exp(-logit));
    return probability >= threshold;
}

std::string FusedTrigramScorer::normalize_url(const std::string &url)
{
    std::string normalized = url;
    normalized.erase(normalized.begin(), std::find_if(normalized.begin(), normalized.end(), [](unsigned char ch)
                                                      { return !std::isspace(ch); }));
    normalized.erase(std::find_if(normalized.rbegin(), normalized.rend(), [](unsigned char ch)
                                  { return !std::isspace(ch); })
                         .base(),
                     normalized.end());

    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch)
                   { return static_cast<char>(std::tolower(ch)); });
    return normalized;
}

std::uint64_t FusedTrigramScorer::trigram_hash(const std::string &text, std::size_t start)
{
    std::uint64_t hash = 0ULL;
    for (std::size_t i = start; i < start + 3 && i < text.size(); ++i)
    {
        hash = (hash * ROLLING_BASE + static_cast<std::uint64_t>(static_cast<unsigned char>(text[i]))) & UINT64_MASK;
    }
    return hash;
}

std::uint64_t FusedTrigramScorer::rolling_update(std::uint64_t prev_hash, unsigned char left_char, unsigned char right_char)
{
    constexpr std::uint64_t base_sq = (ROLLING_BASE * ROLLING_BASE) & UINT64_MASK;
    const std::uint64_t removed = (static_cast<std::uint64_t>(left_char) * base_sq) & UINT64_MASK;
    const std::uint64_t updated = ((prev_hash - removed) & UINT64_MASK) * ROLLING_BASE + static_cast<std::uint64_t>(right_char);
    return updated & UINT64_MASK;
}
