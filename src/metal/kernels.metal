// Metal compute kernels for the frequency separation pipeline. Each kernel
// mirrors one step of src/core/FreqSep.cpp; FreqSepMetal.mm drives them.
// Convention: buffers at indices 0..n-1, then one parameter struct at n.
#include <metal_stdlib>
using namespace metal;

struct Dims {
    int w, h;
};

struct SubInfo {
    int w, h;    // full resolution
    int lw, lh;  // reduced resolution
    int sub;
};

struct BoxInfo {
    int w, h, r;
};

struct ImgInfo {
    int w, h, channels, rowStride, space;
};

struct ScalarInfo {
    int w, h;
    float a, b;
};

struct TexInfo {
    int w, h;
    float texture, t;
    int limit;
};

struct MaskInfo {
    int w, h;
    int hasMask, channels, channelOffset, rowStride;
    int dx, dy, mw, mh;
    float mix;
    int invert;
};

struct SkinInfo {
    int w, h, space;
    float hue0, width, minSat;
};

struct OutInfo {
    int w, h;
    int srcChannels, srcStride, dstChannels, dstStride;
    int view, splitX, lineW, space;
};

constant float kLogOffset = 0.01f;

inline float toWork(float v, int space)
{
    if (space == 1) {
        float lo = log2(kLogOffset), range = log2(1.0f + kLogOffset) - lo;
        return (log2(max(v, 0.0f) + kLogOffset) - lo) / range;
    }
    return v;
}

inline float fromWork(float v, int space)
{
    if (space == 1) {
        float lo = log2(kLogOffset), range = log2(1.0f + kLogOffset) - lo;
        return exp2(v * range + lo) - kLogOffset;
    }
    return v;
}

inline float luma(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

inline float sampleBilinear(device const float* in, int lw, int lh, int x, int y, int sub)
{
    float fx = clamp((x + 0.5f) / sub - 0.5f, 0.0f, float(lw - 1));
    float fy = clamp((y + 0.5f) / sub - 0.5f, 0.0f, float(lh - 1));
    int x0 = int(fx), y0 = int(fy);
    int x1 = min(x0 + 1, lw - 1), y1 = min(y0 + 1, lh - 1);
    float tx = fx - x0, ty = fy - y0;
    float top = in[y0 * lw + x0] + (in[y0 * lw + x1] - in[y0 * lw + x0]) * tx;
    float bot = in[y1 * lw + x0] + (in[y1 * lw + x1] - in[y1 * lw + x0]) * tx;
    return top + (bot - top) * ty;
}

#define GUARD(W, H) \
    if (int(gid.x) >= (W) || int(gid.y) >= (H)) return; \
    const int x = int(gid.x), y = int(gid.y);

kernel void unpack(device const float* src [[buffer(0)]], device float* p0 [[buffer(1)]],
                   device float* p1 [[buffer(2)]], device float* p2 [[buffer(3)]],
                   constant ImgInfo& in [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(in.w, in.h)
    device const float* s = src + y * in.rowStride + x * in.channels;
    int i = y * in.w + x;
    p0[i] = toWork(s[0], in.space);
    p1[i] = toWork(s[1], in.space);
    p2[i] = toWork(s[2], in.space);
}

kernel void downsample(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                       constant SubInfo& s [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.lw, s.lh)
    float sum = 0.0f;
    int n = 0;
    for (int yy = y * s.sub; yy < min(s.h, (y + 1) * s.sub); ++yy)
        for (int xx = x * s.sub; xx < min(s.w, (x + 1) * s.sub); ++xx, ++n) sum += in[yy * s.w + xx];
    out[y * s.lw + x] = sum / n;
}

kernel void upsample(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                     constant SubInfo& s [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.w, s.h)
    out[y * s.w + x] = sampleBilinear(in, s.lw, s.lh, x, y, s.sub);
}

kernel void boxH(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                 constant BoxInfo& b [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(b.w, b.h)
    device const float* row = in + y * b.w;
    float sum = 0.0f;
    for (int i = -b.r; i <= b.r; ++i) sum += row[clamp(x + i, 0, b.w - 1)];
    out[y * b.w + x] = sum / (2 * b.r + 1);
}

kernel void boxV(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                 constant BoxInfo& b [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(b.w, b.h)
    float sum = 0.0f;
    for (int i = -b.r; i <= b.r; ++i) sum += in[clamp(y + i, 0, b.h - 1) * b.w + x];
    out[y * b.w + x] = sum / (2 * b.r + 1);
}

kernel void copyPlane(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                      constant Dims& d [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(d.w, d.h)
    out[y * d.w + x] = in[y * d.w + x];
}

// out = a - b
kernel void subtractPlanes(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                     device float* out [[buffer(2)]], constant Dims& d [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]])
{
    GUARD(d.w, d.h)
    int i = y * d.w + x;
    out[i] = a[i] - b[i];
}

kernel void multiplyPlanes(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                     device float* out [[buffer(2)]], constant Dims& d [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]])
{
    GUARD(d.w, d.h)
    int i = y * d.w + x;
    out[i] = a[i] * b[i];
}

kernel void lumaPlane(device const float* p0 [[buffer(0)]], device const float* p1 [[buffer(1)]],
                      device const float* p2 [[buffer(2)]], device float* out [[buffer(3)]],
                      constant Dims& d [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(d.w, d.h)
    int i = y * d.w + x;
    out[i] = luma(p0[i], p1[i], p2[i]);
}

// a holds mean(I*P) on entry and the slope on exit. s.a = eps.
kernel void guidedCoef(device const float* meanI [[buffer(0)]], device const float* meanII [[buffer(1)]],
                       device const float* meanP [[buffer(2)]], device float* a [[buffer(3)]],
                       device float* b [[buffer(4)]], constant ScalarInfo& s [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.w, s.h)
    int i = y * s.w + x;
    float varI = meanII[i] - meanI[i] * meanI[i];
    float cov = a[i] - meanI[i] * meanP[i];
    float slope = cov / (varI + s.a);
    a[i] = slope;
    b[i] = meanP[i] - slope * meanI[i];
}

struct ApplyInfo {
    int w, h, lw, lh, sub;
    float amount;
};

kernel void guidedApply(device float* P [[buffer(0)]], device const float* I [[buffer(1)]],
                        device const float* a [[buffer(2)]], device const float* b [[buffer(3)]],
                        constant ApplyInfo& s [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.w, s.h)
    int i = y * s.w + x;
    float q = sampleBilinear(a, s.lw, s.lh, x, y, s.sub) * I[i] + sampleBilinear(b, s.lw, s.lh, x, y, s.sub);
    P[i] += (q - P[i]) * s.amount;
}

// p += s.a * (p - base)
kernel void addDetail(device float* p [[buffer(0)]], device const float* base [[buffer(1)]],
                      constant ScalarInfo& s [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.w, s.h)
    int i = y * s.w + x;
    p[i] += s.a * (p[i] - base[i]);
}

// Edge strength of the colour layer luma; s.a = sigmaSep.
kernel void gradEdge(device const float* l [[buffer(0)]], device float* out [[buffer(1)]],
                     constant ScalarInfo& s [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.w, s.h)
    int xl = max(x - 1, 0), xr = min(x + 1, s.w - 1), yd = max(y - 1, 0), yu = min(y + 1, s.h - 1);
    float gx = l[y * s.w + xr] - l[y * s.w + xl];
    float gy = l[yu * s.w + x] - l[yd * s.w + x];
    float contrast = 0.5f * sqrt(gx * gx + gy * gy) * s.a;
    out[y * s.w + x] = clamp(contrast / 0.06f, 0.0f, 1.0f);
}

kernel void textureGain(device float* h0 [[buffer(0)]], device float* h1 [[buffer(1)]],
                        device float* h2 [[buffer(2)]], device const float* edge [[buffer(3)]],
                        constant TexInfo& s [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.w, s.h)
    int i = y * s.w + x;
    float g = s.texture;
    if (s.limit) {
        float hl = luma(h0[i], h1[i], h2[i]);
        if (hl < -1e-6f) {
            float lim = s.t * tanh(hl / s.t) / hl;
            float e = min(1.0f, edge[i] * 1.5f);
            g *= lim + (1.0f - lim) * e;
        }
    }
    h0[i] *= g;
    h1[i] *= g;
    h2[i] *= g;
}

kernel void addPlanes(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                device float* out [[buffer(2)]], constant Dims& d [[buffer(3)]],
                uint2 gid [[thread_position_in_grid]])
{
    GUARD(d.w, d.h)
    int i = y * d.w + x;
    out[i] = a[i] + b[i];
}

kernel void maskInit(device const float* mask [[buffer(0)]], device float* m [[buffer(1)]],
                     constant MaskInfo& s [[buffer(2)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.w, s.h)
    float v;
    if (s.hasMask) {
        int mx = x + s.dx, my = y + s.dy;
        v = (mx >= 0 && my >= 0 && mx < s.mw && my < s.mh)
                ? clamp(mask[my * s.rowStride + mx * s.channels + s.channelOffset], 0.0f, 1.0f)
                : 0.0f;
    } else {
        v = s.invert ? 0.0f : clamp(s.mix, 0.0f, 1.0f);
    }
    m[y * s.w + x] = v;
}

kernel void maskFinish(device float* m [[buffer(0)]], constant MaskInfo& s [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.w, s.h)
    int i = y * s.w + x;
    float v = clamp(m[i], 0.0f, 1.0f);
    m[i] = (s.invert ? 1.0f - v : v) * clamp(s.mix, 0.0f, 1.0f);
}

kernel void skinKey(device const float* l0 [[buffer(0)]], device const float* l1 [[buffer(1)]],
                    device const float* l2 [[buffer(2)]], device float* key [[buffer(3)]],
                    constant SkinInfo& s [[buffer(4)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.w, s.h)
    int i = y * s.w + x;
    float r = fromWork(l0[i], s.space), g = fromWork(l1[i], s.space), b = fromWork(l2[i], s.space);
    float yy = luma(r, g, b);
    float cb = b - yy, cr = r - yy;
    float chroma = sqrt(cb * cb + cr * cr);
    float sat = chroma / max(fabs(yy), 0.02f);
    float diff = atan2(-cb, cr) - s.hue0;
    const float twoPi = 6.28318530718f;
    diff -= twoPi * rint(diff / twoPi);
    float d = diff / s.width;
    float hueW = exp(-d * d * 2.0f);
    float satW = clamp((sat - 0.5f * s.minSat) / s.minSat, 0.0f, 1.0f);
    key[i] = hueW * satW;
}

kernel void writeOutput(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                        device const float* l0 [[buffer(2)]], device const float* l1 [[buffer(3)]],
                        device const float* l2 [[buffer(4)]], device const float* h0 [[buffer(5)]],
                        device const float* h1 [[buffer(6)]], device const float* h2 [[buffer(7)]],
                        device const float* o0 [[buffer(8)]], device const float* o1 [[buffer(9)]],
                        device const float* o2 [[buffer(10)]], device const float* m [[buffer(11)]],
                        constant OutInfo& s [[buffer(12)]], uint2 gid [[thread_position_in_grid]])
{
    GUARD(s.w, s.h)
    int i = y * s.w + x;
    device const float* sp = src + y * s.srcStride + x * s.srcChannels;
    device float* dp = dst + y * s.dstStride + x * s.dstChannels;
    float3 v;
    float3 sv = float3(sp[0], sp[1], sp[2]);
    int view = s.view;
    if (view == 4) {  // split screen
        if (x >= s.splitX && x < s.splitX + s.lineW) view = -1;
        else if (x < s.splitX) view = -2;
        else view = 0;
    }
    switch (view) {
    case -1: v = float3(1.0f); break;
    case -2: v = sv; break;
    case 1: v = float3(fromWork(l0[i], s.space), fromWork(l1[i], s.space), fromWork(l2[i], s.space)); break;
    case 2: v = float3(0.5f) + float3(h0[i], h1[i], h2[i]); break;
    case 3: v = float3(m[i]); break;
    default: {
        float3 o = float3(fromWork(o0[i], s.space), fromWork(o1[i], s.space), fromWork(o2[i], s.space));
        v = sv + (o - sv) * m[i];
        break;
    }
    }
    dp[0] = v.x;
    dp[1] = v.y;
    dp[2] = v.z;
    if (s.dstChannels == 4) dp[3] = s.srcChannels == 4 ? sp[3] : 1.0f;
}
