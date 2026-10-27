// Person segmentation with VNGeneratePersonSegmentationRequest. Each frame is
// segmented on its own (no temporal state), so renders are repeatable and
// frames can come in any order.
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>

#include "PersonMask.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <dispatch/dispatch.h>

namespace sep {

int thumbFactor(int height, PersonQuality q)
{
    // Roughly the resolution each quality level works at internally.
    const int target = q == PersonQuality::Fast ? 360 : q == PersonQuality::Balanced ? 540 : 1080;
    return std::max(1, height / target);
}

void makeThumb(const ImageView& src, int factor, Thumb& out)
{
    const int f = std::max(1, factor);
    out.width = std::max(1, src.width / f);
    out.height = std::max(1, src.height / f);
    out.rgb.assign(static_cast<size_t>(out.width) * out.height * 3, 0.0f);
    const float norm = 1.0f / (f * f);
    // Blocks copy what they capture, so hand them plain values and pointers.
    float* dst = out.rgb.data();
    const int tw = out.width;
    const ImageView s = src;
    dispatch_apply(static_cast<size_t>(out.height), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t ty) {
                       float* o = dst + ty * tw * 3;
                       for (int dy = 0; dy < f; ++dy) {
                           const int y = std::min(s.height - 1, static_cast<int>(ty) * f + dy);
                           const float* row = s.row(y);
                           for (int tx = 0; tx < tw; ++tx)
                               for (int dx = 0; dx < f; ++dx) {
                                   const float* p = row + std::min(s.width - 1, tx * f + dx) * s.channels;
                                   o[tx * 3 + 0] += p[0] * norm;
                                   o[tx * 3 + 1] += p[1] * norm;
                                   o[tx * 3 + 2] += p[2] * norm;
                               }
                       }
                   });
}

uint64_t thumbHash(const Thumb& t)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint32_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(static_cast<uint32_t>(t.width));
    mix(static_cast<uint32_t>(t.height));
    for (float v : t.rgb) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        mix(bits);
    }
    return h;
}

bool personMaskAvailable()
{
    if (@available(macOS 12.0, *)) return true;
    return false;
}

namespace {

// The model expects display-referred pictures. Gamma/log material goes in as
// is; scene linear gets an sRGB-like curve first.
inline uint8_t encode(float v, WorkingSpace space)
{
    v = std::clamp(v, 0.0f, 1.0f);
    if (space == WorkingSpace::Linear)
        v = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    return static_cast<uint8_t>(v * 255.0f + 0.5f);
}

}  // namespace

bool personMask(const Thumb& t, WorkingSpace space, PersonQuality q, int width, int height,
                std::vector<float>& mask, std::string* err)
{
    if (t.width <= 0 || t.height <= 0 || width <= 0 || height <= 0) return false;
    if (@available(macOS 12.0, *)) {
        @autoreleasepool {
            CVPixelBufferRef pb = nullptr;
            NSDictionary* attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
            if (CVPixelBufferCreate(kCFAllocatorDefault, t.width, t.height, kCVPixelFormatType_32BGRA,
                                    (__bridge CFDictionaryRef)attrs, &pb) != kCVReturnSuccess) {
                if (err) *err = "Person mask: could not allocate the input picture.";
                return false;
            }
            CVPixelBufferLockBaseAddress(pb, 0);
            auto* base = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pb));
            const size_t bpr = CVPixelBufferGetBytesPerRow(pb);
            for (int y = 0; y < t.height; ++y) {
                // Vision wants the top row first; the thumbnail starts at the bottom.
                const float* s = t.rgb.data() + static_cast<size_t>(t.height - 1 - y) * t.width * 3;
                uint8_t* d = base + y * bpr;
                for (int x = 0; x < t.width; ++x) {
                    d[x * 4 + 0] = encode(s[x * 3 + 2], space);
                    d[x * 4 + 1] = encode(s[x * 3 + 1], space);
                    d[x * 4 + 2] = encode(s[x * 3 + 0], space);
                    d[x * 4 + 3] = 255;
                }
            }
            CVPixelBufferUnlockBaseAddress(pb, 0);

            VNGeneratePersonSegmentationRequest* req = [VNGeneratePersonSegmentationRequest new];
            req.qualityLevel = q == PersonQuality::Fast       ? VNGeneratePersonSegmentationRequestQualityLevelFast
                               : q == PersonQuality::Balanced ? VNGeneratePersonSegmentationRequestQualityLevelBalanced
                                                              : VNGeneratePersonSegmentationRequestQualityLevelAccurate;
            req.outputPixelFormat = kCVPixelFormatType_OneComponent8;
            VNImageRequestHandler* handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pb options:@{}];
            NSError* e = nil;
            const BOOL ok = [handler performRequests:@[ req ] error:&e];
            CVPixelBufferRelease(pb);
            VNPixelBufferObservation* obs = ok ? req.results.firstObject : nil;
            if (!obs) {
                if (err)
                    *err = std::string("Person mask failed: ") +
                           (e ? e.localizedDescription.UTF8String : "no result from Vision");
                return false;
            }

            CVPixelBufferRef mb = obs.pixelBuffer;
            CVPixelBufferLockBaseAddress(mb, kCVPixelBufferLock_ReadOnly);
            const auto* m = static_cast<const uint8_t*>(CVPixelBufferGetBaseAddress(mb));
            const int mw = static_cast<int>(CVPixelBufferGetWidth(mb));
            const int mh = static_cast<int>(CVPixelBufferGetHeight(mb));
            const size_t mbpr = CVPixelBufferGetBytesPerRow(mb);

            // Bilinear upsample to full size, flipping back to bottom-first rows.
            mask.resize(static_cast<size_t>(width) * height);
            float* out = mask.data();
            const float sx = static_cast<float>(mw) / width, sy = static_cast<float>(mh) / height;
            dispatch_apply(static_cast<size_t>(height), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                           ^(size_t y) {
                               const int top = height - 1 - static_cast<int>(y);
                               const float fy = std::clamp((top + 0.5f) * sy - 0.5f, 0.0f, float(mh - 1));
                               const int y0 = static_cast<int>(fy), y1 = std::min(y0 + 1, mh - 1);
                               const float wy = fy - y0;
                               const uint8_t* r0 = m + y0 * mbpr;
                               const uint8_t* r1 = m + y1 * mbpr;
                               float* o = out + y * width;
                               for (int x = 0; x < width; ++x) {
                                   const float fx = std::clamp((x + 0.5f) * sx - 0.5f, 0.0f, float(mw - 1));
                                   const int x0 = static_cast<int>(fx), x1 = std::min(x0 + 1, mw - 1);
                                   const float wx = fx - x0;
                                   const float a = r0[x0] + (r0[x1] - r0[x0]) * wx;
                                   const float b = r1[x0] + (r1[x1] - r1[x0]) * wx;
                                   o[x] = (a + (b - a) * wy) * (1.0f / 255.0f);
                               }
                           });
            CVPixelBufferUnlockBaseAddress(mb, kCVPixelBufferLock_ReadOnly);
            return true;
        }
    }
    if (err) *err = "The person mask needs macOS 12 or newer.";
    return false;
}

}  // namespace sep
