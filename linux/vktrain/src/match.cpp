#include "match.h"

#include <algorithm>
#include <cstring>
#include <thread>

namespace vkt {

namespace {
constexpr float kNone = 3.0e38f;

struct PC {
    uint32_t aOffset, nA, bOffset, nB, outOffset;
};
}  // namespace

Matches finishMatches(const NN& ab, const NN& ba, double ratio) {
    Matches m;
    const double r2 = ratio * ratio;   // on squared distances: d1 < r² · d2
    for (uint32_t i = 0; i < ab.idx.size(); i++) {
        uint32_t j = ab.idx[i];
        if (j == UINT32_MAX || j >= ba.idx.size() || ba.idx[j] != i) continue;
        if (ab.d2[i] < kNone && !(double(ab.d1[i]) < r2 * double(ab.d2[i]))) continue;
        if (ba.d2[j] < kNone && !(double(ba.d1[j]) < r2 * double(ba.d2[j]))) continue;
        m.emplace_back(i, j);
    }
    return m;
}

NN cpuNearest(const DescSet& a, const DescSet& b, unsigned threads) {
    NN out;
    out.idx.assign(a.n, UINT32_MAX);
    out.d1.assign(a.n, kNone);
    out.d2.assign(a.n, kNone);
    std::vector<float> nb(b.n);
    for (uint32_t j = 0; j < b.n; j++) {
        float s = 0;
        for (int k = 0; k < 128; k++) s += float(b.d[size_t(j) * 128 + size_t(k)]) * float(b.d[size_t(j) * 128 + size_t(k)]);
        nb[j] = s;
    }
    auto work = [&](uint32_t lo, uint32_t hi) {
        std::vector<float> q(128);
        for (uint32_t i = lo; i < hi; i++) {
            float na = 0;
            for (int k = 0; k < 128; k++) {
                q[size_t(k)] = float(a.d[size_t(i) * 128 + size_t(k)]);
                na += q[size_t(k)] * q[size_t(k)];
            }
            float best = kNone, second = kNone;
            uint32_t bi = UINT32_MAX;
            for (uint32_t j = 0; j < b.n; j++) {
                const uint8_t* p = &b.d[size_t(j) * 128];
                float dp = 0;
                for (int k = 0; k < 128; k++) dp += q[size_t(k)] * float(p[k]);
                float d = na + nb[j] - 2.0f * dp;
                if (d < best) {
                    second = best;
                    best = d;
                    bi = j;
                } else if (d < second) {
                    second = d;
                }
            }
            out.idx[i] = bi;
            out.d1[i] = best;
            out.d2[i] = second;
        }
    };
    threads = std::max(1u, threads);
    std::vector<std::thread> pool;
    uint32_t chunk = (a.n + threads - 1) / threads;
    for (unsigned t = 0; t < threads; t++) {
        uint32_t lo = std::min(a.n, t * chunk), hi = std::min(a.n, lo + chunk);
        if (lo < hi) pool.emplace_back(work, lo, hi);
    }
    for (auto& t : pool) t.join();
    return out;
}

GpuMatcher::~GpuMatcher() {
    ctx_.destroy(desc_);
    ctx_.destroy(norm_);
    ctx_.destroy(out_);
}

void GpuMatcher::upload(const std::vector<DescSet>& images) {
    offset_.clear();
    count_.clear();
    uint32_t total = 0;
    for (const auto& im : images) {
        offset_.push_back(total);
        count_.push_back(im.n);
        total += im.n;
    }
    std::vector<float> f(size_t(std::max(total, 1u)) * 128, 0.0f), n2(std::max(total, 1u), 0.0f);
    for (size_t i = 0; i < images.size(); i++) {
        const auto& im = images[i];
        for (uint32_t j = 0; j < im.n; j++) {
            float s = 0;
            for (int k = 0; k < 128; k++) {
                float v = float(im.d[size_t(j) * 128 + size_t(k)]);
                f[(size_t(offset_[i]) + j) * 128 + size_t(k)] = v;
                s += v * v;
            }
            n2[offset_[i] + j] = s;
        }
    }
    ctx_.ensure(desc_, f.size() * sizeof(float));
    ctx_.ensure(norm_, n2.size() * sizeof(float));
    ctx_.upload(desc_, f.data(), f.size() * sizeof(float));
    ctx_.upload(norm_, n2.data(), n2.size() * sizeof(float));
}

void GpuMatcher::run(const std::vector<std::pair<int, int>>& pairs, std::vector<NN>& ab, std::vector<NN>& ba) {
    uint32_t slots = 0;
    std::vector<uint32_t> outAB(pairs.size()), outBA(pairs.size());
    for (size_t p = 0; p < pairs.size(); p++) {
        outAB[p] = slots;
        slots += count_[size_t(pairs[p].first)];
        outBA[p] = slots;
        slots += count_[size_t(pairs[p].second)];
    }
    ctx_.ensure(out_, size_t(std::max(slots, 1u)) * 3 * sizeof(uint32_t));
    ctx_.begin();
    for (size_t p = 0; p < pairs.size(); p++) {
        uint32_t a = uint32_t(pairs[p].first), b = uint32_t(pairs[p].second);
        for (int dir = 0; dir < 2; dir++) {
            uint32_t q = dir == 0 ? a : b, t = dir == 0 ? b : a;
            PC pc{offset_[q], count_[q], offset_[t], count_[t], dir == 0 ? outAB[p] : outBA[p]};
            if (pc.nA == 0) continue;
            ctx_.dispatch("match_nn", {&desc_, &norm_, &out_}, &pc, sizeof(pc), (pc.nA + 63) / 64);
        }
    }
    ctx_.submitWait();
    std::vector<uint32_t> raw(size_t(slots) * 3);
    if (slots) ctx_.download(out_, raw.data(), raw.size() * sizeof(uint32_t));
    ab.assign(pairs.size(), NN());
    ba.assign(pairs.size(), NN());
    auto unpack = [&](uint32_t start, uint32_t n, NN& nn) {
        nn.idx.resize(n);
        nn.d1.resize(n);
        nn.d2.resize(n);
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t* r = &raw[(size_t(start) + i) * 3];
            nn.idx[i] = r[0];
            std::memcpy(&nn.d1[i], &r[1], 4);
            std::memcpy(&nn.d2[i], &r[2], 4);
        }
    };
    for (size_t p = 0; p < pairs.size(); p++) {
        unpack(outAB[p], count_[size_t(pairs[p].first)], ab[p]);
        unpack(outBA[p], count_[size_t(pairs[p].second)], ba[p]);
    }
}

}  // namespace vkt
