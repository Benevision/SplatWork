// SIFT descriptor matching on any Vulkan GPU (docs/dev/GPU_PROCESSING.md, stage 1): brute-force nearest
// neighbours both ways (shaders/match_nn.comp), then Lowe's ratio test and a mutual check on the host — and an
// exact CPU reference doing the same, so the two give identical matches (the distances are exact integers).
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "vk.h"

namespace vkt {

// One image's descriptors: n × 128 bytes (SIFT, 0…255).
struct DescSet {
    uint32_t n = 0;
    std::vector<uint8_t> d;
};

// Nearest neighbours of every descriptor of A in B: index (UINT32_MAX when B is empty), dist² of the nearest and
// of the second nearest (FLT_MAX-like 3e38 when there is none).
struct NN {
    std::vector<uint32_t> idx;
    std::vector<float> d1, d2;
};

using Matches = std::vector<std::pair<uint32_t, uint32_t>>;

// Mutual nearest neighbours that pass the ratio test (nearest < ratio × second nearest, on distances) both ways.
Matches finishMatches(const NN& ab, const NN& ba, double ratio);

// The exact CPU reference (float dot products of byte-valued vectors are exact integers), on `threads` threads.
NN cpuNearest(const DescSet& a, const DescSet& b, unsigned threads);

class GpuMatcher {
public:
    explicit GpuMatcher(Context& ctx) : ctx_(ctx) {}
    ~GpuMatcher();
    // All images' descriptors to the GPU once (pairs then index into them).
    void upload(const std::vector<DescSet>& images);
    // Nearest neighbours for every pair, both directions, in one submission.
    void run(const std::vector<std::pair<int, int>>& pairs, std::vector<NN>& ab, std::vector<NN>& ba);

private:
    Context& ctx_;
    Buffer desc_, norm_, out_;
    std::vector<uint32_t> offset_, count_;
};

}  // namespace vkt
