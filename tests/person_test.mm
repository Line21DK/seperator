// Checks the AI person mask on a real portrait (NASA, public domain): the
// face must be in the mask and the wall behind her must not. Also checks that
// the Metal thumbnail and CPU-side mask path agree with the CPU pipeline.
// Skips (exit 0) when the OS or machine cannot run person segmentation.
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/core/FreqSep.h"
#include "../src/metal/FreqSepMetal.h"
#include "../src/vision/PersonMask.h"

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// RGBA float, row 0 = bottom row (like an OFX image).
bool loadImage(const char* path, std::vector<float>& px, int& w, int& h)
{
    NSURL* url = [NSURL fileURLWithPath:@(path)];
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!src) return false;
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    if (!img) return false;
    w = static_cast<int>(CGImageGetWidth(img));
    h = static_cast<int>(CGImageGetHeight(img));
    std::vector<uint8_t> rgba(size_t(w) * h * 4);
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(rgba.data(), w, h, 8, size_t(w) * 4, cs,
                                             kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CGImageRelease(img);
    px.resize(size_t(w) * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 4; ++c)
                px[(size_t(h - 1 - y) * w + x) * 4 + c] = rgba[(size_t(y) * w + x) * 4 + c] / 255.0f;
    return true;
}

// Mean of the mask over a rectangle given in top-down picture coordinates.
float regionMean(const std::vector<float>& m, int w, int h, int x0, int y0, int x1, int y1)
{
    double sum = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) sum += m[size_t(h - 1 - y) * w + x];
    return static_cast<float>(sum / ((x1 - x0) * (y1 - y0)));
}

void metalChecks(std::vector<float>& px, int w, int h, const std::vector<float>& mask)
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::printf("no Metal device, skipping Metal checks\n");
        return;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    const size_t bytes = px.size() * sizeof(float);
    id<MTLBuffer> srcBuf = [device newBufferWithBytes:px.data() length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    sep::MetalImage s{(__bridge void*)srcBuf, w, h, 4, w * 4};
    sep::MetalImage d{(__bridge void*)dstBuf, w, h, 4, w * 4};

    // Thumbnail on the GPU equals the CPU one.
    sep::ImageView sv{px.data(), w, h, 4, w * 4};
    sep::Thumb cpu, gpu;
    sep::makeThumb(sv, 3, cpu);
    std::string err;
    bool ok = sep::thumbnailMetal((__bridge void*)queue, s, 3, gpu.rgb, gpu.width, gpu.height, &err);
    float maxDiff = 0;
    if (ok && gpu.rgb.size() == cpu.rgb.size())
        for (size_t i = 0; i < cpu.rgb.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(cpu.rgb[i] - gpu.rgb[i]));
    std::printf("thumbnail %dx%d, max diff %.2e %s\n", gpu.width, gpu.height, maxDiff, err.c_str());
    check(ok && gpu.width == cpu.width && gpu.height == cpu.height && maxDiff < 1e-5f, "Metal thumbnail matches CPU");

    // The pipeline with a CPU-side person mask matches the CPU pipeline.
    sep::Params p;
    p.skinKey = true;
    p.maskSoftness = 4;
    p.colorSmooth = 24;
    std::vector<float> ref(px.size());
    sep::ImageView rv{ref.data(), w, h, 4, w * 4};
    sep::process(sv, rv, mask.data(), p);
    sep::MetalMask mm;
    mm.host = mask.data();
    mm.width = w;
    mm.height = h;
    ok = sep::processMetal((__bridge void*)queue, s, d, &mm, p, true, &err);
    const float* out = static_cast<const float*>(dstBuf.contents);
    maxDiff = 0;
    for (size_t i = 0; i < ref.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(ref[i] - out[i]));
    std::printf("person-masked render, max diff %.2e %s\n", maxDiff, err.c_str());
    check(ok && maxDiff < 5e-3f, "Metal render with person mask matches CPU");
}

}  // namespace

int main(int argc, char** argv)
{
    @autoreleasepool {
        if (argc < 2) {
            std::printf("usage: person_test portrait.jpg\n");
            return 1;
        }
        if (!sep::personMaskAvailable()) {
            std::printf("person segmentation needs macOS 12, skipping\n");
            return 0;
        }
        std::vector<float> px;
        int w = 0, h = 0;
        if (!loadImage(argv[1], px, w, h)) {
            std::printf("could not load %s\n", argv[1]);
            return 1;
        }

        sep::ImageView sv{px.data(), w, h, 4, w * 4};
        sep::Thumb thumb;
        sep::makeThumb(sv, sep::thumbFactor(h, sep::PersonQuality::Balanced), thumb);
        std::vector<float> mask;
        std::string err;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = sep::personMask(thumb, sep::WorkingSpace::AsIs, sep::PersonQuality::Balanced, w, h, mask, &err);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) {
            // CI runners have no GPU or Neural Engine; Vision cannot run there.
            if (!MTLCreateSystemDefaultDevice()) {
                std::printf("%s (no GPU), skipping\n", err.c_str());
                return 0;
            }
            std::printf("FAIL %s\n", err.c_str());
            return 1;
        }
        std::printf("person mask %dx%d in %.1f ms\n", w, h, ms);

        const float face = regionMean(mask, w, h, 190, 80, 260, 160);
        const float suit = regionMean(mask, w, h, 60, 330, 120, 450);
        const float wall = regionMean(mask, w, h, 305, 20, 345, 120);
        const float corner = regionMean(mask, w, h, 440, 0, 512, 40);
        std::printf("face %.2f, suit %.2f, wall %.2f, corner %.2f\n", face, suit, wall, corner);
        check(face > 0.8f, "face is in the mask");
        check(suit > 0.8f, "suit is in the mask");
        check(wall < 0.2f, "wall behind her is not");
        check(corner < 0.2f, "top right corner is not");

        // Accurate and Fast also run.
        for (auto q : {sep::PersonQuality::Fast, sep::PersonQuality::Accurate}) {
            sep::Thumb t;
            sep::makeThumb(sv, sep::thumbFactor(h, q), t);
            std::vector<float> m2;
            const bool ok2 = sep::personMask(t, sep::WorkingSpace::AsIs, q, w, h, m2, &err);
            check(ok2 && regionMean(m2, w, h, 190, 80, 260, 160) > 0.8f,
                  q == sep::PersonQuality::Fast ? "Fast quality finds the face" : "Accurate quality finds the face");
        }

        metalChecks(px, w, h, mask);
        std::printf(failures ? "%d failures\n" : "all good\n", failures);
        return failures ? 1 : 0;
    }
}
