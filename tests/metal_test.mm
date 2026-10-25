// Runs the Metal pipeline and the CPU core on the same images and checks
// that they agree. Skips (exit 0) when no Metal device is available.
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../src/core/FreqSep.h"
#include "../src/metal/FreqSepMetal.h"

namespace {

int failures = 0;

std::vector<float> makeTest(int w, int h)
{
    std::vector<float> px(size_t(w) * h * 4);
    std::mt19937 rng(1);
    std::normal_distribution<float> fine(0.0f, 0.01f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float base = x < w / 2 ? 0.25f : 0.75f;
            // a warm, skin-like disc so the skin key has something to find
            float dx = x - w * 0.3f, dy = y - h * 0.5f;
            bool disc = dx * dx + dy * dy < (h * 0.3f) * (h * 0.3f);
            float blotch = 0.04f * std::sin(x * 0.15f) * std::sin(y * 0.18f);
            float n = fine(rng);
            float* p = &px[(size_t(y) * w + x) * 4];
            float v = (disc ? 0.6f : base) + blotch;
            p[0] = v + n;
            p[1] = (disc ? 0.75f : 0.8f) * v + n;
            p[2] = (disc ? 0.6f : 0.6f) * v + n;
            p[3] = x < w * 2 / 3 ? 1.0f : 0.0f;  // alpha doubles as a mask
        }
    return px;
}

struct Case {
    const char* name;
    sep::Params p;
    bool alphaMask;
};

}  // namespace

int main()
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::printf("no Metal device, skipping\n");
            return 0;
        }
        std::printf("Metal device: %s\n", device.name.UTF8String);
        id<MTLCommandQueue> queue = [device newCommandQueue];

        const int W = 960, H = 540;
        std::vector<float> src = makeTest(W, H);
        const size_t bytes = src.size() * sizeof(float);
        id<MTLBuffer> srcBuf = [device newBufferWithBytes:src.data() length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dstBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

        std::vector<Case> cases;
        cases.push_back({"defaults", sep::Params{}, false});
        {
            sep::Params p;
            p.colorSmooth = 30.0f;
            p.edgeProtect = 0.2f;
            p.blemish = 0.6f;
            p.midtoneDetail = 0.4f;
            p.postSharpen = 0.8f;
            p.texture = 1.3f;
            p.skinKey = true;
            p.maskSoftness = 6.0f;
            p.mix = 0.8f;
            cases.push_back({"everything + alpha mask", p, true});
            p.space = sep::WorkingSpace::Linear;
            p.invertMask = true;
            cases.push_back({"linear + inverted mask", p, true});
            for (int v = 1; v <= 4; ++v) {
                sep::Params q = p;
                q.space = sep::WorkingSpace::AsIs;
                q.invertMask = false;
                q.view = static_cast<sep::View>(v);
                static const char* names[] = {"", "view colour layer", "view texture", "view mask", "view split"};
                cases.push_back({names[v], q, true});
            }
        }

        for (const Case& c : cases) {
            std::vector<float> cpu(src.size());
            std::vector<float> maskPlane;
            if (c.alphaMask) {
                maskPlane.resize(size_t(W) * H);
                for (size_t i = 0; i < maskPlane.size(); ++i) maskPlane[i] = src[i * 4 + 3];
            }
            sep::ImageView sv{src.data(), W, H, 4, W * 4};
            sep::ImageView cv{cpu.data(), W, H, 4, W * 4};
            sep::process(sv, cv, c.alphaMask ? maskPlane.data() : nullptr, c.p);

            sep::MetalImage ms{(__bridge void*)srcBuf, W, H, 4, W * 4};
            sep::MetalImage md{(__bridge void*)dstBuf, W, H, 4, W * 4};
            sep::MetalMask mm{(__bridge void*)srcBuf, 4, 3, W * 4, 0, 0, W, H};
            std::string err;
            bool ok = sep::processMetal((__bridge void*)queue, ms, md, c.alphaMask ? &mm : nullptr, c.p, true, &err);
            if (!ok) {
                std::printf("FAIL  %s: %s\n", c.name, err.c_str());
                ++failures;
                continue;
            }
            const float* gpu = static_cast<const float*>(dstBuf.contents);
            float maxDiff = 0.0f;
            double meanDiff = 0.0;
            for (size_t i = 0; i < cpu.size(); ++i) {
                float d = std::fabs(cpu[i] - gpu[i]);
                if (!std::isfinite(gpu[i])) d = INFINITY;
                maxDiff = std::max(maxDiff, d);
                meanDiff += d;
            }
            meanDiff /= cpu.size();
            bool pass = maxDiff < 5e-3f && meanDiff < 2e-4;
            std::printf("%s  %-26s max %.2e  mean %.2e\n", pass ? "ok  " : "FAIL", c.name, maxDiff, meanDiff);
            if (!pass) ++failures;
        }

        // 4K timing (first run compiles nothing new; buffers get allocated).
        {
            const int BW = 3840, BH = 2160;
            const size_t bb = size_t(BW) * BH * 4 * sizeof(float);
            id<MTLBuffer> s = [device newBufferWithLength:bb options:MTLResourceStorageModePrivate];
            id<MTLBuffer> d = [device newBufferWithLength:bb options:MTLResourceStorageModePrivate];
            sep::MetalImage ms{(__bridge void*)s, BW, BH, 4, BW * 4};
            sep::MetalImage md{(__bridge void*)d, BW, BH, 4, BW * 4};
            sep::Params p;
            p.skinKey = true;
            p.blemish = 0.5f;
            std::string err;
            sep::processMetal((__bridge void*)queue, ms, md, nullptr, p, true, &err);
            auto t0 = std::chrono::steady_clock::now();
            const int frames = 5;
            for (int i = 0; i < frames; ++i) sep::processMetal((__bridge void*)queue, ms, md, nullptr, p, true, &err);
            double ms4k = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / frames;
            std::printf("4K frame on GPU: %.1f ms\n", ms4k);
        }

        std::printf(failures ? "%d failure(s)\n" : "all passed\n", failures);
        return failures ? 1 : 0;
    }
}
