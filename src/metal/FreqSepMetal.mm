// Metal driver for the frequency separation pipeline. Step for step the same
// as sep::process in src/core/FreqSep.cpp; tests/metal_test.mm compares both.
#import <Metal/Metal.h>

#include "FreqSepMetal.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "../core/BlurPlan.h"
#include "MetalKernelsSource.h"  // generated: const char* kMetalKernelsSource

namespace sep {
namespace {

struct Dims { int w, h; };
struct SubInfo { int w, h, lw, lh, sub; };
struct BoxInfo { int w, h, r; };
struct ImgInfo { int w, h, channels, rowStride, space; };
struct ScalarInfo { int w, h; float a, b; };
struct TexInfo { int w, h; float texture, t; int limit; };
struct MaskInfo {
    int w, h;
    int hasMask, channels, channelOffset, rowStride;
    int dx, dy, mw, mh;
    float mix;
    int invert;
};
struct SkinInfo { int w, h, space; float hue0, width, minSat; };
struct OutInfo { int w, h, srcChannels, srcStride, dstChannels, dstStride, view, splitX, lineW, space; };
struct ApplyInfo { int w, h, lw, lh, sub; float amount; };
struct ThumbInfo { int w, h, channels, rowStride, factor, tw, th; };

// Compiled pipelines per device, scratch buffers per command queue. Buffers
// are reused frame to frame; work on one queue executes in order, so reuse
// is safe without waiting.
struct DeviceState {
    std::map<std::string, id<MTLComputePipelineState>> pipelines;
};
struct QueueState {
    std::map<std::string, id<MTLBuffer>> buffers;
};

std::mutex gMutex;
std::map<void*, DeviceState> gDevices;
std::map<void*, QueueState> gQueues;

bool loadPipelines(id<MTLDevice> device, DeviceState& ds, std::string* err)
{
    if (!ds.pipelines.empty()) return true;
    NSError* e = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    id<MTLLibrary> lib = [device newLibraryWithSource:@(kMetalKernelsSource) options:options error:&e];
    if (!lib) {
        if (err) *err = std::string("Metal compile failed: ") + e.localizedDescription.UTF8String;
        return false;
    }
    for (NSString* name in lib.functionNames) {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:fn error:&e];
        if (!ps) {
            if (err) *err = std::string("Metal pipeline failed: ") + e.localizedDescription.UTF8String;
            ds.pipelines.clear();
            return false;
        }
        ds.pipelines[name.UTF8String] = ps;
    }
    return true;
}

class Encoder {
public:
    Encoder(id<MTLDevice> device, id<MTLComputeCommandEncoder> enc, DeviceState& ds, QueueState& qs)
        : _device(device), _enc(enc), _ds(ds), _qs(qs)
    {
    }

    // Scratch buffer by name, at least `floats` long.
    id<MTLBuffer> buf(const std::string& name, size_t floats)
    {
        id<MTLBuffer> __strong& b = _qs.buffers[name];
        const NSUInteger bytes = std::max<size_t>(floats, 1) * sizeof(float);
        if (!b || b.length < bytes) b = [_device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        return b;
    }

    template <class P>
    void run(const char* kernel, std::initializer_list<id<MTLBuffer>> buffers, const P& params, int w, int h)
    {
        id<MTLComputePipelineState> ps = _ds.pipelines.at(kernel);
        [_enc setComputePipelineState:ps];
        NSUInteger i = 0;
        for (id<MTLBuffer> b : buffers) [_enc setBuffer:b offset:0 atIndex:i++];
        [_enc setBytes:&params length:sizeof(P) atIndex:i];
        MTLSize group = MTLSizeMake(16, 16, 1);
        MTLSize groups = MTLSizeMake((w + 15) / 16, (h + 15) / 16, 1);
        [_enc dispatchThreadgroups:groups threadsPerThreadgroup:group];
    }

    void gauss(id<MTLBuffer> data, id<MTLBuffer> tmp, int w, int h, float sigma)
    {
        if (sigma < 0.5f) return;
        int radii[3];
        boxRadiiForGauss(sigma, radii);
        for (int r : radii) {
            if (r < 1) continue;
            BoxInfo b{w, h, r};
            run("boxH", {data, tmp}, b, w, h);
            run("boxV", {tmp, data}, b, w, h);
        }
    }

    // Same as smooth() in FreqSep.cpp.
    void smooth(id<MTLBuffer> data, int w, int h, float sigma)
    {
        if (sigma < 0.5f) return;
        const int sub = smoothFactor(sigma);
        if (sub == 1) {
            gauss(data, buf("tmp", size_t(w) * h), w, h, sigma);
            return;
        }
        SubInfo s{w, h, (w + sub - 1) / sub, (h + sub - 1) / sub, sub};
        id<MTLBuffer> small = buf("smoothSmall", size_t(s.lw) * s.lh);
        id<MTLBuffer> stmp = buf("smoothTmp", size_t(s.lw) * s.lh);
        run("downsample", {data, small}, s, s.lw, s.lh);
        gauss(small, stmp, s.lw, s.lh, sigma / sub);
        run("upsample", {small, data}, s, w, h);
    }

private:
    id<MTLDevice> _device;
    id<MTLComputeCommandEncoder> _enc;
    DeviceState& _ds;
    QueueState& _qs;
};

}  // namespace

bool processMetal(void* commandQueue, const MetalImage& src, const MetalImage& dst, const MetalMask* mask,
                  const Params& p, bool wait, std::string* err)
{
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLDevice> device = queue.device;
        const int w = src.width, h = src.height;
        if (w <= 0 || h <= 0) return true;
        const size_t N = size_t(w) * h;
        const float scale = h / 1080.0f;
        const int space = static_cast<int>(p.space);

        std::lock_guard<std::mutex> lock(gMutex);
        DeviceState& ds = gDevices[(__bridge void*)device];
        if (!loadPipelines(device, ds, err)) return false;
        QueueState& qs = gQueues[commandQueue];

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        cmd.label = @"Seperator";
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        Encoder E(device, enc, ds, qs);
        const Dims full{w, h};

        id<MTLBuffer> srcBuf = (__bridge id<MTLBuffer>)src.buffer;
        id<MTLBuffer> dstBuf = (__bridge id<MTLBuffer>)dst.buffer;
        id<MTLBuffer> in[3] = {E.buf("in0", N), E.buf("in1", N), E.buf("in2", N)};
        id<MTLBuffer> low[3] = {E.buf("low0", N), E.buf("low1", N), E.buf("low2", N)};
        id<MTLBuffer> high[3] = {E.buf("high0", N), E.buf("high1", N), E.buf("high2", N)};

        // Unpack into planar working-space buffers.
        ImgInfo ii{w, h, src.channels, src.rowStride, space};
        E.run("unpack", {srcBuf, in[0], in[1], in[2]}, ii, w, h);

        // 1. Split.
        const float sigmaSep = sigmaSeparation(p.separation, scale);
        for (int k = 0; k < 3; ++k) {
            E.run("copyPlane", {in[k], low[k]}, full, w, h);
            E.smooth(low[k], w, h, sigmaSep);
            E.run("subtractPlanes", {in[k], low[k], high[k]}, full, w, h);
        }

        id<MTLBuffer> lowOrig[3] = {nil, nil, nil};
        if (p.skinKey || p.blemish > 0.0f)
            for (int k = 0; k < 3; ++k) {
                lowOrig[k] = E.buf("lowOrig" + std::to_string(k), N);
                E.run("copyPlane", {low[k], lowOrig[k]}, full, w, h);
            }

        // 2. Guided filter on the colour layer, coefficients at reduced resolution.
        if (p.colorAmount > 0.0f && p.colorSmooth > 0.0f) {
            const float sigmaG = sigmaGuided(p.colorSmooth, scale);
            const float eps = guidedEps(p.edgeProtect);
            const int sub = guidedFactor(sigmaG, sigmaSep);
            SubInfo s{w, h, (w + sub - 1) / sub, (h + sub - 1) / sub, sub};
            const size_t LN = size_t(s.lw) * s.lh;
            const Dims lowDims{s.lw, s.lh};
            const float lsigma = sigmaG / sub;
            id<MTLBuffer> ltmp = E.buf("gTmp", LN);
            id<MTLBuffer> I = E.buf("gI", N), lI = E.buf("gLI", LN), lII = E.buf("gLII", LN);
            id<MTLBuffer> meanI = E.buf("gMeanI", LN), lP = E.buf("gLP", LN), meanP = E.buf("gMeanP", LN);
            id<MTLBuffer> a = E.buf("gA", LN), b = E.buf("gB", LN);

            E.run("lumaPlane", {low[0], low[1], low[2], I}, full, w, h);
            E.run("downsample", {I, lI}, s, s.lw, s.lh);
            E.run("multiplyPlanes", {lI, lI, lII}, lowDims, s.lw, s.lh);
            E.run("copyPlane", {lI, meanI}, lowDims, s.lw, s.lh);
            E.gauss(meanI, ltmp, s.lw, s.lh, lsigma);
            E.gauss(lII, ltmp, s.lw, s.lh, lsigma);
            ApplyInfo ai{w, h, s.lw, s.lh, sub, std::clamp(p.colorAmount, 0.0f, 1.0f)};
            ScalarInfo ci{s.lw, s.lh, eps, 0.0f};
            for (int k = 0; k < 3; ++k) {
                E.run("downsample", {low[k], lP}, s, s.lw, s.lh);
                E.run("copyPlane", {lP, meanP}, lowDims, s.lw, s.lh);
                E.gauss(meanP, ltmp, s.lw, s.lh, lsigma);
                E.run("multiplyPlanes", {lI, lP, a}, lowDims, s.lw, s.lh);
                E.gauss(a, ltmp, s.lw, s.lh, lsigma);
                E.run("guidedCoef", {meanI, lII, meanP, a, b}, ci, s.lw, s.lh);
                E.gauss(a, ltmp, s.lw, s.lh, lsigma);
                E.gauss(b, ltmp, s.lw, s.lh, lsigma);
                E.run("guidedApply", {low[k], I, a, b}, ai, w, h);
            }
        }

        // 3. Midtone detail.
        if (p.midtoneDetail != 0.0f) {
            id<MTLBuffer> base = E.buf("base", N);
            ScalarInfo si{w, h, p.midtoneDetail, 0.0f};
            for (int k = 0; k < 3; ++k) {
                E.run("copyPlane", {low[k], base}, full, w, h);
                E.smooth(base, w, h, sigmaSep * 3.0f);
                E.run("addDetail", {low[k], base}, si, w, h);
            }
        }

        // 4. Texture gain and blemish limiting.
        {
            const bool limit = p.blemish > 0.0f;
            id<MTLBuffer> edge = E.buf("edge", limit ? N : 1);
            if (limit) {
                id<MTLBuffer> l = E.buf("edgeLuma", N);
                E.run("lumaPlane", {lowOrig[0], lowOrig[1], lowOrig[2], l}, full, w, h);
                ScalarInfo gi{w, h, sigmaSep, 0.0f};
                E.run("gradEdge", {l, edge}, gi, w, h);
                E.smooth(edge, w, h, sigmaSep);
            }
            TexInfo ti{w, h, p.texture, blemishThreshold(p.blemish), limit ? 1 : 0};
            E.run("textureGain", {high[0], high[1], high[2], edge}, ti, w, h);
        }

        // 5. Recombine (into the `in` planes), with optional post sharpen.
        auto& out = in;
        for (int k = 0; k < 3; ++k) {
            E.run("addPlanes", {low[k], high[k], out[k]}, full, w, h);
            if (p.postSharpen > 0.0f) {
                id<MTLBuffer> soft = E.buf("soft", N);
                E.run("copyPlane", {out[k], soft}, full, w, h);
                E.smooth(soft, w, h, sharpenSigma(scale));
                ScalarInfo si{w, h, p.postSharpen, 0.0f};
                E.run("addDetail", {out[k], soft}, si, w, h);
            }
        }

        // 6. Strength map.
        id<MTLBuffer> m = E.buf("mask", N);
        const bool hasMask = mask && (mask->buffer || mask->host);
        MaskInfo mi{w, h, hasMask ? 1 : 0, 1, 0, 0, 0, 0, 0, 0, p.mix, p.invertMask ? 1 : 0};
        id<MTLBuffer> maskBuf = m;  // any valid buffer when there is no mask
        if (hasMask && mask->host) {
            // A fresh buffer per frame: the previous frame may still be reading its own.
            maskBuf = [device newBufferWithBytes:mask->host
                                          length:size_t(mask->width) * mask->height * sizeof(float)
                                         options:MTLResourceStorageModeShared];
            mi.channels = 1;
            mi.channelOffset = 0;
            mi.rowStride = mask->width;
            mi.dx = mask->dx;
            mi.dy = mask->dy;
            mi.mw = mask->width;
            mi.mh = mask->height;
        } else if (hasMask) {
            maskBuf = (__bridge id<MTLBuffer>)mask->buffer;
            mi.channels = mask->channels;
            mi.channelOffset = mask->channelOffset;
            mi.rowStride = mask->rowStride;
            mi.dx = mask->dx;
            mi.dy = mask->dy;
            mi.mw = mask->width;
            mi.mh = mask->height;
        }
        E.run("maskInit", {maskBuf, m}, mi, w, h);
        if (hasMask) {
            E.smooth(m, w, h, p.maskSoftness * scale);
            E.run("maskFinish", {m}, mi, w, h);
        }
        if (p.skinKey) {
            id<MTLBuffer> key = E.buf("key", N);
            const float deg = 3.14159265f / 180.0f;
            SkinInfo si{w, h, space, p.skinHue * deg, std::max(1.0f, p.skinHueWidth) * deg, std::max(1e-3f, p.skinMinSat)};
            E.run("skinKey", {lowOrig[0], lowOrig[1], lowOrig[2], key}, si, w, h);
            E.smooth(key, w, h, skinKeySigma(scale));
            E.run("multiplyPlanes", {m, key, m}, full, w, h);
        }

        // 7. Output.
        OutInfo oi{w, h, src.channels, src.rowStride, dst.channels, dst.rowStride, static_cast<int>(p.view),
                   static_cast<int>(std::clamp(p.splitPosition, 0.0f, 1.0f) * w),
                   std::max(1, static_cast<int>(scale + 0.5f)), space};
        E.run("writeOutput",
              {srcBuf, dstBuf, low[0], low[1], low[2], high[0], high[1], high[2], out[0], out[1], out[2], m}, oi, w,
              h);

        [enc endEncoding];
        [cmd commit];
        if (wait) {
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) {
                if (err) *err = std::string("Metal command buffer failed: ") + cmd.error.localizedDescription.UTF8String;
                return false;
            }
        }
        return true;
    }
}

bool thumbnailMetal(void* commandQueue, const MetalImage& src, int factor, std::vector<float>& rgb, int& width,
                    int& height, std::string* err)
{
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLDevice> device = queue.device;
        const int f = std::max(1, factor);
        ThumbInfo ti{src.width, src.height, src.channels, src.rowStride, f, std::max(1, src.width / f),
                     std::max(1, src.height / f)};
        const size_t floats = size_t(ti.tw) * ti.th * 3;
        id<MTLBuffer> out = [device newBufferWithLength:floats * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd;
        {
            std::lock_guard<std::mutex> lock(gMutex);
            DeviceState& ds = gDevices[(__bridge void*)device];
            if (!loadPipelines(device, ds, err)) return false;
            QueueState& qs = gQueues[commandQueue];
            cmd = [queue commandBuffer];
            cmd.label = @"Seperator thumbnail";
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            Encoder E(device, enc, ds, qs);
            E.run("thumbnail", {(__bridge id<MTLBuffer>)src.buffer, out}, ti, ti.tw, ti.th);
            [enc endEncoding];
            [cmd commit];
        }
        [cmd waitUntilCompleted];
        if (cmd.status == MTLCommandBufferStatusError) {
            if (err) *err = std::string("Metal thumbnail failed: ") + cmd.error.localizedDescription.UTF8String;
            return false;
        }
        rgb.assign(static_cast<const float*>(out.contents), static_cast<const float*>(out.contents) + floats);
        width = ti.tw;
        height = ti.th;
        return true;
    }
}

}  // namespace sep
