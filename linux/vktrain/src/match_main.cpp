// splatwork-vk-match: SIFT descriptor matching on any Vulkan GPU (docs/dev/GPU_PROCESSING.md, stage 1).
//
//   splatwork-vk-match --info [--device N]
//   splatwork-vk-match bench [--keypoints 4096] [--images 16] [--window 4] [--ratio 0.8] [--threads 0]
//                            [--seed 1] [--no-cpu] [--device N]
//       Synthetic SIFT-like descriptors (scene points seen by neighbouring images, with noise, plus distractors),
//       sequential pairs (each image with the next `window`); times the GPU against the exact CPU reference on
//       `threads` threads (0 = all cores) and checks the matches are identical. Prints a summary and, last, one
//       JSON line {"type": "bench", …}.
//   splatwork-vk-match match --desc DIR --pairs FILE [--ratio 0.8] [--cpu] [--threads 0] [--device N]
//       DIR/<name>.sift: u32 n (little-endian) then n × 128 bytes. FILE: one "nameA nameB" pair per line. Prints a
//       JSON line per pair {"type": "pair", "a", "b", "matches": [[i, j], …]} and, last, {"type": "done", …}.
//       --cpu: the CPU reference instead of the GPU (for comparisons).
// Exit codes: 0 done (bench: identical matches), 1 error / bench mismatch, 2 invalid arguments.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "match.h"
#include "nlohmann/json.hpp"

using namespace vkt;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

double secondsSince(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

// A SIFT-like descriptor: non-negative, L2-normalised, clipped at 0.2, renormalised, ×512, saturated to 255.
void siftLike(std::mt19937& rng, float* v) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (int k = 0; k < 128; k++) v[k] = std::fabs(n(rng)) * (k % 8 == 0 ? 2.0f : 1.0f);
}

void quantise(const float* v, uint8_t* out) {
    double s = 0;
    for (int k = 0; k < 128; k++) s += double(v[k]) * v[k];
    s = std::sqrt(std::max(s, 1e-12));
    float t[128];
    double s2 = 0;
    for (int k = 0; k < 128; k++) {
        t[k] = std::min(float(v[k] / s), 0.2f);
        s2 += double(t[k]) * t[k];
    }
    s2 = std::sqrt(std::max(s2, 1e-12));
    for (int k = 0; k < 128; k++) out[k] = uint8_t(std::min(255.0, std::round(512.0 * t[k] / s2)));
}

// Images along a path: image i sees scene points [i·step, i·step + shared) (60 % of its keypoints) with noise, plus
// 40 % unique distractors, in random order.
std::vector<DescSet> synthetic(uint32_t images, uint32_t kps, uint32_t window, uint32_t seed) {
    std::mt19937 rng(seed);
    uint32_t shared = kps * 6 / 10, step = std::max(1u, shared / (window + 1));
    uint32_t points = step * images + shared;
    std::vector<float> scene(size_t(points) * 128);
    for (uint32_t p = 0; p < points; p++) siftLike(rng, &scene[size_t(p) * 128]);
    std::normal_distribution<float> noise(0.0f, 0.12f);
    std::vector<DescSet> out(images);
    for (uint32_t i = 0; i < images; i++) {
        DescSet& d = out[i];
        d.n = kps;
        d.d.resize(size_t(kps) * 128);
        std::vector<uint32_t> order(kps);
        for (uint32_t k = 0; k < kps; k++) order[k] = k;
        std::shuffle(order.begin(), order.end(), rng);
        for (uint32_t k = 0; k < kps; k++) {
            float v[128];
            if (k < shared) {
                const float* s = &scene[(size_t(i) * step + k) * 128];
                for (int c = 0; c < 128; c++) v[c] = std::max(0.0f, s[c] * (1.0f + noise(rng)));
            } else {
                siftLike(rng, v);
            }
            quantise(v, &d.d[size_t(order[k]) * 128]);
        }
    }
    return out;
}

std::vector<std::pair<int, int>> sequentialPairs(uint32_t images, uint32_t window) {
    std::vector<std::pair<int, int>> p;
    for (uint32_t i = 0; i < images; i++)
        for (uint32_t j = i + 1; j <= std::min(images - 1, i + window); j++) p.emplace_back(int(i), int(j));
    return p;
}

unsigned threadsOr(unsigned t) {
    return t ? t : std::max(1u, std::thread::hardware_concurrency());
}

int bench(int argc, char** argv, int i, ContextOptions co) {
    uint32_t kps = 4096, images = 16, window = 4, seed = 1;
    double ratio = 0.8;
    unsigned threads = 0;
    bool cpu = true;
    for (; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--keypoints") kps = uint32_t(std::stoul(next()));
        else if (a == "--images") images = uint32_t(std::stoul(next()));
        else if (a == "--window") window = uint32_t(std::stoul(next()));
        else if (a == "--ratio") ratio = std::stod(next());
        else if (a == "--threads") threads = unsigned(std::stoul(next()));
        else if (a == "--seed") seed = uint32_t(std::stoul(next()));
        else if (a == "--no-cpu") cpu = false;
        else if (a == "--device") co.deviceIndex = std::stoi(next());
        else throw std::invalid_argument("unknown argument " + a);
    }
    images = std::max(images, 2u);
    threads = threadsOr(threads);
    std::printf("building %u images × %u descriptors…\n", images, kps);
    std::fflush(stdout);
    auto data = synthetic(images, kps, window, seed);
    auto pairs = sequentialPairs(images, window);
    Context ctx(co);
    GpuMatcher gm(ctx);
    gm.upload(data);
    std::vector<NN> ab, ba;
    gm.run({pairs.front()}, ab, ba);                 // warm-up: pipeline creation, first submission
    auto t0 = Clock::now();
    gm.run(pairs, ab, ba);
    double gpuSec = secondsSince(t0);
    size_t matches = 0;
    std::vector<Matches> gpuM(pairs.size());
    for (size_t p = 0; p < pairs.size(); p++) {
        gpuM[p] = finishMatches(ab[p], ba[p], ratio);
        matches += gpuM[p].size();
    }
    double gpuMs = 1e3 * gpuSec / double(pairs.size());
    std::printf("GPU  %s: %.2f ms per pair (%zu pairs, %u × %u descriptors), %.0f matches per pair\n",
                ctx.info.name.c_str(), gpuMs, pairs.size(), kps, kps, double(matches) / double(pairs.size()));
    json j = {{"type", "bench"}, {"device", ctx.info.name}, {"deviceType", ctx.info.deviceType},
              {"driver", ctx.info.driver}, {"keypoints", kps}, {"images", images}, {"pairs", pairs.size()},
              {"gpuMsPerPair", gpuMs}, {"matchesPerPair", double(matches) / double(pairs.size())}};
    bool identical = true;
    if (cpu) {
        // the reference on a few pairs is enough to time it and to check every pair it does
        size_t nCpu = std::min<size_t>(pairs.size(), 4);
        auto t1 = Clock::now();
        size_t diff = 0;
        for (size_t p = 0; p < nCpu; p++) {
            NN cab = cpuNearest(data[size_t(pairs[p].first)], data[size_t(pairs[p].second)], threads);
            NN cba = cpuNearest(data[size_t(pairs[p].second)], data[size_t(pairs[p].first)], threads);
            Matches cm = finishMatches(cab, cba, ratio);
            if (cm != gpuM[p] || cab.idx != ab[p].idx || cba.idx != ba[p].idx || cab.d1 != ab[p].d1 ||
                cab.d2 != ab[p].d2)
                diff++;
        }
        double cpuMs = 1e3 * secondsSince(t1) / double(nCpu);
        identical = diff == 0;
        std::printf("CPU  %u threads: %.1f ms per pair (the exact reference loop, %zu pairs; not SIMD-optimised — "
                    "match_bench.py run times OpenCV) → GPU %.1f×; matches %s\n",
                    threads, cpuMs, nCpu, cpuMs / gpuMs, identical ? "identical" : "DIFFER");
        j.update({{"cpuThreads", threads}, {"cpuMsPerPair", cpuMs}, {"speedup", cpuMs / gpuMs},
                  {"checkedPairs", nCpu}, {"identical", identical}});
    }
    std::printf("%s\n", j.dump().c_str());
    return identical ? 0 : 1;
}

DescSet readSift(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    DescSet d;
    f.read(reinterpret_cast<char*>(&d.n), 4);
    d.d.resize(size_t(d.n) * 128);
    f.read(reinterpret_cast<char*>(d.d.data()), std::streamsize(d.d.size()));
    if (!f) throw std::runtime_error(p.string() + " is truncated");
    return d;
}

int match(int argc, char** argv, int i, ContextOptions co) {
    std::string dir, pairsFile;
    double ratio = 0.8;
    bool useCpu = false;
    unsigned threads = 0;
    for (; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--desc") dir = next();
        else if (a == "--pairs") pairsFile = next();
        else if (a == "--ratio") ratio = std::stod(next());
        else if (a == "--cpu") useCpu = true;
        else if (a == "--threads") threads = unsigned(std::stoul(next()));
        else if (a == "--device") co.deviceIndex = std::stoi(next());
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (dir.empty() || pairsFile.empty()) throw std::invalid_argument("match needs --desc DIR and --pairs FILE");
    std::vector<std::string> names;
    std::vector<std::pair<int, int>> pairs;
    auto id = [&](const std::string& n) {
        for (size_t k = 0; k < names.size(); k++)
            if (names[k] == n) return int(k);
        names.push_back(n);
        return int(names.size() - 1);
    };
    std::ifstream pf(pairsFile);
    std::string a, b;
    while (pf >> a >> b) pairs.emplace_back(id(a), id(b));
    std::vector<DescSet> data;
    for (const auto& n : names) data.push_back(readSift(std::filesystem::path(dir) / (n + ".sift")));
    std::vector<Matches> out(pairs.size());
    auto t0 = Clock::now();
    std::string device = "cpu";
    if (useCpu) {
        unsigned th = threadsOr(threads);
        for (size_t p = 0; p < pairs.size(); p++)
            out[p] = finishMatches(cpuNearest(data[size_t(pairs[p].first)], data[size_t(pairs[p].second)], th),
                                   cpuNearest(data[size_t(pairs[p].second)], data[size_t(pairs[p].first)], th), ratio);
        device = "cpu (" + std::to_string(th) + " threads)";
    } else {
        Context ctx(co);
        device = ctx.info.name;
        GpuMatcher gm(ctx);
        gm.upload(data);
        std::vector<NN> ab, ba;
        gm.run(pairs, ab, ba);
        for (size_t p = 0; p < pairs.size(); p++) out[p] = finishMatches(ab[p], ba[p], ratio);
    }
    double sec = secondsSince(t0);
    for (size_t p = 0; p < pairs.size(); p++) {
        json m = json::array();
        for (auto& [x, y] : out[p]) m.push_back({x, y});
        std::printf("%s\n", json({{"type", "pair"}, {"a", names[size_t(pairs[p].first)]},
                                  {"b", names[size_t(pairs[p].second)]}, {"matches", m}}).dump().c_str());
    }
    std::printf("%s\n", json({{"type", "done"}, {"device", device}, {"pairs", pairs.size()}, {"seconds", sec},
                              {"msPerPair", pairs.empty() ? 0.0 : 1e3 * sec / double(pairs.size())}}).dump().c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    ContextOptions co;
    try {
        if (argc >= 2 && std::string(argv[1]) == "bench") return bench(argc, argv, 2, co);
        if (argc >= 2 && std::string(argv[1]) == "match") return match(argc, argv, 2, co);
        bool info = false;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--info") info = true;
            else if (a == "--device" && i + 1 < argc) co.deviceIndex = std::stoi(argv[++i]);
            else {
                std::fprintf(stderr, "unknown argument %s\n", a.c_str());
                return 2;
            }
        }
        if (info) {
            Context ctx(co);
            std::printf("%s\n", json({{"device", ctx.info.name}, {"deviceType", ctx.info.deviceType},
                                      {"driver", ctx.info.driver}, {"memoryMB", ctx.info.deviceLocalBytes >> 20}}).dump().c_str());
            return 0;
        }
        std::printf("usage: splatwork-vk-match --info | bench [--keypoints N] [--images N] [--window N] [--threads N] "
                    "[--no-cpu] [--device N] | match --desc DIR --pairs FILE [--ratio R] [--cpu] [--device N]\n");
        return 2;
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "splatwork-vk-match: %s\n", e.what());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "splatwork-vk-match: %s\n", e.what());
        std::printf("%s\n", json({{"type", "error"}, {"message", e.what()}}).dump().c_str());
        return 1;
    }
}
