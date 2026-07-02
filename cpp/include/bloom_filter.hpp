#pragma once

#include <cstdint>
#include <string>
#include <vector>

class BloomFilter
{
public:
    BloomFilter(std::uint64_t bit_count, std::uint32_t num_hashes);
    void add(const std::string &key);
    bool contains(const std::string &key) const;

private:
    std::uint64_t bit_count_;
    std::uint32_t num_hashes_;
    std::vector<std::uint64_t> bits_;

    std::uint64_t hash64(const std::string &key, std::uint64_t seed) const;
    void set_bit(std::uint64_t idx);
    bool test_bit(std::uint64_t idx) const;
};
