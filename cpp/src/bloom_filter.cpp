#include "bloom_filter.hpp"

#include <functional>

BloomFilter::BloomFilter(std::uint64_t bit_count, std::uint32_t num_hashes)
    : bit_count_(bit_count),
      num_hashes_(num_hashes),
      bits_((bit_count + 63) / 64, 0ULL) {}

void BloomFilter::add(const std::string &key)
{
    const std::uint64_t h1 = hash64(key, 0x9E3779B185EBCA87ULL);
    const std::uint64_t h2 = hash64(key, 0xC2B2AE3D27D4EB4FULL);
    for (std::uint32_t i = 0; i < num_hashes_; ++i)
    {
        const std::uint64_t idx = (h1 + i * h2) % bit_count_;
        set_bit(idx);
    }
}

bool BloomFilter::contains(const std::string &key) const
{
    const std::uint64_t h1 = hash64(key, 0x9E3779B185EBCA87ULL);
    const std::uint64_t h2 = hash64(key, 0xC2B2AE3D27D4EB4FULL);
    for (std::uint32_t i = 0; i < num_hashes_; ++i)
    {
        const std::uint64_t idx = (h1 + i * h2) % bit_count_;
        if (!test_bit(idx))
        {
            return false;
        }
    }
    return true;
}

std::uint64_t BloomFilter::hash64(const std::string &key, std::uint64_t seed) const
{
    return static_cast<std::uint64_t>(std::hash<std::string>{}(key) ^ seed);
}

void BloomFilter::set_bit(std::uint64_t idx)
{
    bits_[idx / 64] |= (1ULL << (idx % 64));
}

bool BloomFilter::test_bit(std::uint64_t idx) const
{
    return (bits_[idx / 64] & (1ULL << (idx % 64))) != 0ULL;
}
