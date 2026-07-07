#include "bloom_filter.hpp"

#include <cstring>
#include <stdexcept>

namespace
{
    constexpr std::uint64_t PRIME64_1 = 11400714785074694791ULL;
    constexpr std::uint64_t PRIME64_2 = 14029467366897019727ULL;
    constexpr std::uint64_t PRIME64_3 = 1609587929392839161ULL;
    constexpr std::uint64_t PRIME64_4 = 9650029242287828579ULL;
    constexpr std::uint64_t PRIME64_5 = 2870177450012600261ULL;

    std::uint64_t rotl64(std::uint64_t value, int shift)
    {
        return (value << shift) | (value >> (64 - shift));
    }

    std::uint64_t read_u64(const unsigned char *ptr)
    {
        std::uint64_t value;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    }

    std::uint32_t read_u32(const unsigned char *ptr)
    {
        std::uint32_t value;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    }

    std::uint64_t avalanche(std::uint64_t hash)
    {
        hash ^= hash >> 33;
        hash *= PRIME64_2;
        hash ^= hash >> 29;
        hash *= PRIME64_3;
        hash ^= hash >> 32;
        return hash;
    }

    std::uint64_t xxhash64(const void *input, std::size_t length, std::uint64_t seed)
    {
        const auto *data = static_cast<const unsigned char *>(input);
        const auto *const end = data + length;

        std::uint64_t hash;

        if (length >= 32)
        {
            std::uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
            std::uint64_t v2 = seed + PRIME64_2;
            std::uint64_t v3 = seed + 0ULL;
            std::uint64_t v4 = seed - PRIME64_1;

            const auto *const limit = end - 32;
            do
            {
                v1 += read_u64(data) * PRIME64_2;
                v1 = rotl64(v1, 31);
                v1 *= PRIME64_1;
                data += 8;

                v2 += read_u64(data) * PRIME64_2;
                v2 = rotl64(v2, 31);
                v2 *= PRIME64_1;
                data += 8;

                v3 += read_u64(data) * PRIME64_2;
                v3 = rotl64(v3, 31);
                v3 *= PRIME64_1;
                data += 8;

                v4 += read_u64(data) * PRIME64_2;
                v4 = rotl64(v4, 31);
                v4 *= PRIME64_1;
                data += 8;
            } while (data <= limit);

            hash = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

            auto mix = [](std::uint64_t accumulator, std::uint64_t lane) {
                lane *= PRIME64_2;
                lane = rotl64(lane, 31);
                lane *= PRIME64_1;
                accumulator ^= lane;
                accumulator = accumulator * PRIME64_1 + PRIME64_4;
                return accumulator;
            };

            hash = mix(hash, v1);
            hash = mix(hash, v2);
            hash = mix(hash, v3);
            hash = mix(hash, v4);
        }
        else
        {
            hash = seed + PRIME64_5;
        }

        hash += static_cast<std::uint64_t>(length);

        while (data + 8 <= end)
        {
            std::uint64_t lane = read_u64(data);
            lane *= PRIME64_2;
            lane = rotl64(lane, 31);
            lane *= PRIME64_1;
            hash ^= lane;
            hash = rotl64(hash, 27) * PRIME64_1 + PRIME64_4;
            data += 8;
        }

        if (data + 4 <= end)
        {
            hash ^= static_cast<std::uint64_t>(read_u32(data)) * PRIME64_1;
            hash = rotl64(hash, 23) * PRIME64_2 + PRIME64_3;
            data += 4;
        }

        while (data < end)
        {
            hash ^= static_cast<std::uint64_t>(*data) * PRIME64_5;
            hash = rotl64(hash, 11) * PRIME64_1;
            ++data;
        }

        return avalanche(hash);
    }
}

BloomFilter::BloomFilter(std::uint64_t bit_count, std::uint32_t num_hashes)
    : bit_count_(bit_count),
      num_hashes_(num_hashes),
      bits_((bit_count + 63) / 64, 0ULL)
{
    if (bit_count_ == 0)
    {
        throw std::invalid_argument("bit_count must be greater than zero");
    }
    if (num_hashes_ == 0)
    {
        throw std::invalid_argument("num_hashes must be greater than zero");
    }
}

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
    return xxhash64(key.data(), key.size(), seed);
}

void BloomFilter::set_bit(std::uint64_t idx)
{
    bits_[idx / 64] |= (1ULL << (idx % 64));
}

bool BloomFilter::test_bit(std::uint64_t idx) const
{
    return (bits_[idx / 64] & (1ULL << (idx % 64))) != 0ULL;
}
