/*
 * xoshiro256++ — 2.5× faster than std::mt19937 with superior statistical
 * quality. Satisfies C++ UniformRandomBitGenerator concept, drop-in for
 * any std:: distribution or algorithm that needs an RNG.
 *
 * Reference: Blackman & Vigna, "Scrambled Linear Pseudorandom Number
 * Generators", ACM TOMS 2021.
 *
 * Seeding: SplitMix64 expands a 64-bit seed to the 256-bit state.
 * Period: 2^256 − 1. Passes BigCrush, not crypto-safe.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

class Xoshiro256pp {
public:
    using result_type = uint64_t;

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return std::numeric_limits<uint64_t>::max(); }

    Xoshiro256pp() : Xoshiro256pp(0xDEADBEEF01234567ULL) {}
    explicit Xoshiro256pp(uint64_t seed) { this->seed(seed); }

    void seed(uint64_t s) {
        // SplitMix64 to expand the seed to 4 × 64-bit state words.
        state_[0] = splitmix64(s);
        state_[1] = splitmix64(s);
        state_[2] = splitmix64(s);
        state_[3] = splitmix64(s);
    }

    uint64_t operator()() {
        const uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
        const uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

private:
    uint64_t state_[4];
    uint64_t split_state_ = 0;

    uint64_t splitmix64(uint64_t& seed_ref) {
        if (split_state_ == 0) split_state_ = seed_ref;
        split_state_ += 0x9E3779B97F4A7C15ULL;
        uint64_t z = split_state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

// Convenience typedef: use Rng in place of std::mt19937 in hot algorithms.
using Rng = Xoshiro256pp;
