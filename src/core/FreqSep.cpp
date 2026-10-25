#include "FreqSep.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>

namespace sep {
namespace {

using Plane = std::vector<float>;

// Runs fn(begin, end) over [0, n) split across threads.
void parallelFor(int n, int threads, const std::function<void(int, int)>& fn)
{
    threads = std::max(1, std::min(threads, n));
    if (threads == 1) {
        fn(0, n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        int b = static_cast<int>(static_cast<long long>(n) * t / threads);
        int e = static_cast<int>(static_cast<long long>(n) * (t + 1) / threads);
        pool.emplace_back(fn, b, e);
    }
    for (auto& th : pool) th.join();
}

struct Ctx {
    int w, h, threads;
    std::size_t size() const { return static_cast<std::size_t>(w) * h; }
};

// Radii of three box filters approximating a gaussian of the given sigma.
void boxRadiiForGauss(float sigma, int radii[3])
{
    const int n = 3;
    float wIdeal = std::sqrt(12.0f * sigma * sigma / n + 1.0f);
    int wl = static_cast<int>(std::floor(wIdeal));
    if (wl % 2 == 0) --wl;
    int wu = wl + 2;
    float mIdeal = (12.0f * sigma * sigma - n * wl * wl - 4.0f * n * wl - 3.0f * n) / (-4.0f * wl - 4.0f);
    int m = static_cast<int>(std::lround(mIdeal));
    for (int i = 0; i < n; ++i) radii[i] = ((i < m ? wl : wu) - 1) / 2;
}

void boxH(const Ctx& c, const float* in, float* out, int r)
{
    parallelFor(c.h, c.threads, [&](int y0, int y1) {
        const int w = c.w;
        const float inv = 1.0f / (2 * r + 1);
        for (int y = y0; y < y1; ++y) {
            const float* src = in + static_cast<std::size_t>(y) * w;
            float* dst = out + static_cast<std::size_t>(y) * w;
            auto at = [&](int x) { return src[std::clamp(x, 0, w - 1)]; };
            double sum = 0.0;
            for (int i = -r; i <= r; ++i) sum += at(i);
            for (int x = 0; x < w; ++x) {
                dst[x] = static_cast<float>(sum * inv);
                sum += at(x + r + 1) - at(x - r);
            }
        }
    });
}

// Vertical box filter. Each thread owns a band of columns and walks down the
// image with one running sum per column, so memory access stays row-wise.
void boxV(const Ctx& c, const float* in, float* out, int r)
{
    parallelFor(c.w, c.threads, [&](int x0, int x1) {
        const int w = c.w, h = c.h, n = x1 - x0;
        const float inv = 1.0f / (2 * r + 1);
        std::vector<double> sum(n, 0.0);
        auto row = [&](int y) { return in + static_cast<std::size_t>(std::clamp(y, 0, h - 1)) * w + x0; };
        for (int i = -r; i <= r; ++i) {
            const float* s = row(i);
            for (int k = 0; k < n; ++k) sum[k] += s[k];
        }
        for (int y = 0; y < h; ++y) {
            float* d = out + static_cast<std::size_t>(y) * w + x0;
            const float* add = row(y + r + 1);
            const float* sub = row(y - r);
            for (int k = 0; k < n; ++k) {
                d[k] = static_cast<float>(sum[k] * inv);
                sum[k] += add[k] - sub[k];
            }
        }
    });
}

// In-place gaussian blur (three box passes each way). tmp must be image sized.
void gauss(const Ctx& c, float* data, float sigma, Plane& tmp)
{
    if (sigma < 0.5f) return;
    int radii[3];
    boxRadiiForGauss(sigma, radii);
    for (int r : radii) {
        if (r < 1) continue;
        boxH(c, data, tmp.data(), r);
        boxV(c, tmp.data(), data, r);
    }
}

Plane blurred(const Ctx& c, const Plane& in, float sigma, Plane& tmp)
{
    Plane out = in;
    gauss(c, out.data(), sigma, tmp);
    return out;
}

// Box-averages a full resolution plane down by an integer factor.
Plane downsample(const Ctx& c, const Ctx& lc, const Plane& in, int sub)
{
    Plane out(lc.size());
    parallelFor(lc.h, c.threads, [&](int y0, int y1) {
        for (int ly = y0; ly < y1; ++ly)
            for (int lx = 0; lx < lc.w; ++lx) {
                float sum = 0.0f;
                int n = 0;
                for (int y = ly * sub; y < std::min(c.h, (ly + 1) * sub); ++y)
                    for (int x = lx * sub; x < std::min(c.w, (lx + 1) * sub); ++x, ++n)
                        sum += in[static_cast<std::size_t>(y) * c.w + x];
                out[static_cast<std::size_t>(ly) * lc.w + lx] = sum / n;
            }
    });
    return out;
}

// Bilinear sample of a downsampled plane at full resolution pixel (x, y).
inline float sampleBilinear(const Ctx& lc, const Plane& in, int x, int y, int sub)
{
    float fx = std::clamp((x + 0.5f) / sub - 0.5f, 0.0f, static_cast<float>(lc.w - 1));
    float fy = std::clamp((y + 0.5f) / sub - 0.5f, 0.0f, static_cast<float>(lc.h - 1));
    int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
    int x1 = std::min(x0 + 1, lc.w - 1), y1 = std::min(y0 + 1, lc.h - 1);
    float tx = fx - x0, ty = fy - y0;
    const float* r0 = &in[static_cast<std::size_t>(y0) * lc.w];
    const float* r1 = &in[static_cast<std::size_t>(y1) * lc.w];
    float top = r0[x0] + (r0[x1] - r0[x0]) * tx;
    float bot = r1[x0] + (r1[x1] - r1[x0]) * tx;
    return top + (bot - top) * ty;
}

void upsample(const Ctx& c, const Ctx& lc, const Plane& in, int sub, Plane& out)
{
    parallelFor(c.h, c.threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < c.w; ++x) out[static_cast<std::size_t>(y) * c.w + x] = sampleBilinear(lc, in, x, y, sub);
    });
}

// Element-wise helper over whole planes.
template <class F>
void forEach(const Ctx& c, F f)
{
    parallelFor(static_cast<int>(c.h), c.threads, [&](int y0, int y1) {
        std::size_t b = static_cast<std::size_t>(y0) * c.w, e = static_cast<std::size_t>(y1) * c.w;
        for (std::size_t i = b; i < e; ++i) f(i);
    });
}

const float kLogOffset = 0.01f;
const float kLogLo = std::log2(kLogOffset);
const float kLogRange = std::log2(1.0f + kLogOffset) - kLogLo;

inline float toWork(float v, WorkingSpace s)
{
    if (s == WorkingSpace::Linear) return (std::log2(std::max(v, 0.0f) + kLogOffset) - kLogLo) / kLogRange;
    return v;
}

inline float fromWork(float v, WorkingSpace s)
{
    if (s == WorkingSpace::Linear) return std::exp2(v * kLogRange + kLogLo) - kLogOffset;
    return v;
}

inline float luma(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

}  // namespace

void process(const ImageView& src, const ImageView& dst, const float* mask, const Params& p, int threads)
{
    if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const Ctx c{src.width, src.height, threads};
    if (c.w <= 0 || c.h <= 0) return;
    const std::size_t N = c.size();
    const float scale = c.h / 1080.0f;
    const WorkingSpace ws = p.space;

    // Unpack into planar working-space buffers.
    Plane in[3], tmp(N);
    for (auto& pl : in) pl.resize(N);
    parallelFor(c.h, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const float* s = src.row(y);
            std::size_t o = static_cast<std::size_t>(y) * c.w;
            for (int x = 0; x < c.w; ++x, s += src.channels)
                for (int k = 0; k < 3; ++k) in[k][o + x] = toWork(s[k], ws);
        }
    });

    // 1. Split: colour layer = blur, texture layer = remainder.
    const float sigmaSep = std::max(0.5f, p.separation * scale);
    Plane low[3], high[3];
    // Large blurs run at reduced resolution and are upsampled; the texture
    // layer is defined as the remainder, so recombination stays lossless.
    const int subSep = std::clamp(static_cast<int>(sigmaSep / 3.0f), 1, 4);
    const Ctx sc{(c.w + subSep - 1) / subSep, (c.h + subSep - 1) / subSep, threads};
    for (int k = 0; k < 3; ++k) {
        if (subSep == 1) {
            low[k] = blurred(c, in[k], sigmaSep, tmp);
        } else {
            Plane small = downsample(c, sc, in[k], subSep), stmp(sc.size());
            gauss(sc, small.data(), sigmaSep / subSep, stmp);
            low[k].resize(N);
            upsample(c, sc, small, subSep, low[k]);
        }
        high[k].resize(N);
        forEach(c, [&](std::size_t i) { high[k][i] = in[k][i] - low[k][i]; });
        Plane().swap(in[k]);
    }

    // Unsmoothed colour layer, used by the skin key and the blemish edge guard.
    Plane lowOrig[3];
    if (p.skinKey || p.blemish > 0.0f)
        for (int k = 0; k < 3; ++k) lowOrig[k] = low[k];

    // 2. Smooth the colour layer with a luma-guided guided filter: evens out
    //    blotches and redness while keeping real edges (eyes, lips, jaw).
    //    The colour layer is already low-pass, so the filter coefficients are
    //    solved at reduced resolution ("fast guided filter") and upsampled.
    if (p.colorAmount > 0.0f && p.colorSmooth > 0.0f) {
        const float sigmaG = std::max(0.5f, p.colorSmooth * scale * 0.5f);
        const float eps = std::pow(10.0f, -2.0f - 3.0f * std::clamp(p.edgeProtect, 0.0f, 1.0f));
        const int sub = std::clamp(static_cast<int>(std::min(sigmaG / 3.0f, sigmaSep)), 1, 8);
        const Ctx lc{(c.w + sub - 1) / sub, (c.h + sub - 1) / sub, threads};
        const std::size_t LN = lc.size();
        const float lsigma = sigmaG / sub;
        Plane ltmp(LN);

        Plane I(N);
        forEach(c, [&](std::size_t i) { I[i] = luma(low[0][i], low[1][i], low[2][i]); });
        Plane lI = downsample(c, lc, I, sub), lII(LN);
        forEach(lc, [&](std::size_t i) { lII[i] = lI[i] * lI[i]; });
        Plane meanI = blurred(lc, lI, lsigma, ltmp);
        gauss(lc, lII.data(), lsigma, ltmp);  // lII becomes mean(I*I)
        const float amt = std::clamp(p.colorAmount, 0.0f, 1.0f);
        for (int k = 0; k < 3; ++k) {
            Plane& P = low[k];
            Plane lP = downsample(c, lc, P, sub);
            Plane meanP = blurred(lc, lP, lsigma, ltmp);
            Plane a(LN), b(LN);
            forEach(lc, [&](std::size_t i) { a[i] = lI[i] * lP[i]; });
            gauss(lc, a.data(), lsigma, ltmp);  // a = mean(I*P)
            forEach(lc, [&](std::size_t i) {
                float varI = lII[i] - meanI[i] * meanI[i];
                float cov = a[i] - meanI[i] * meanP[i];
                a[i] = cov / (varI + eps);
                b[i] = meanP[i] - a[i] * meanI[i];
            });
            gauss(lc, a.data(), lsigma, ltmp);
            gauss(lc, b.data(), lsigma, ltmp);
            parallelFor(c.h, threads, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const std::size_t o = static_cast<std::size_t>(y) * c.w;
                    for (int x = 0; x < c.w; ++x) {
                        float q = sampleBilinear(lc, a, x, y, sub) * I[o + x] + sampleBilinear(lc, b, x, y, sub);
                        P[o + x] += (q - P[o + x]) * amt;
                    }
                }
            });
        }
    }

    // 3. Midtone detail: contrast of the band just below the texture layer.
    if (p.midtoneDetail != 0.0f) {
        for (int k = 0; k < 3; ++k) {
            Plane base = blurred(c, low[k], sigmaSep * 3.0f, tmp);
            forEach(c, [&](std::size_t i) { low[k][i] += p.midtoneDetail * (low[k][i] - base[i]); });
        }
    }

    // 4. Texture gain and blemish limiting. Dark dips in the texture layer
    //    (spots, pimples) are compressed with tanh, while fine pores and
    //    highlights pass untouched. Real edges (lash lines, lips, jaw) also
    //    make big dips, so the limiter fades out where the colour layer has a
    //    strong gradient.
    {
        const float t = 0.12f * (1.0f - p.blemish) * (1.0f - p.blemish) + 0.012f;
        const bool limit = p.blemish > 0.0f;
        Plane edge;
        if (limit) {
            edge.resize(N);
            forEach(c, [&](std::size_t i) { edge[i] = luma(lowOrig[0][i], lowOrig[1][i], lowOrig[2][i]); });
            Plane grad(N);
            parallelFor(c.h, threads, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < c.w; ++x) {
                        auto at = [&](int xx, int yy) {
                            return edge[static_cast<std::size_t>(std::clamp(yy, 0, c.h - 1)) * c.w + std::clamp(xx, 0, c.w - 1)];
                        };
                        float gx = at(x + 1, y) - at(x - 1, y), gy = at(x, y + 1) - at(x, y - 1);
                        // contrast across roughly one separation radius
                        float contrast = 0.5f * std::sqrt(gx * gx + gy * gy) * sigmaSep;
                        grad[static_cast<std::size_t>(y) * c.w + x] = std::clamp(contrast / 0.06f, 0.0f, 1.0f);
                    }
            });
            gauss(c, grad.data(), sigmaSep, tmp);
            edge.swap(grad);
        }
        forEach(c, [&](std::size_t i) {
            float g = p.texture;
            if (limit) {
                float hl = luma(high[0][i], high[1][i], high[2][i]);
                if (hl < -1e-6f) {
                    float lim = t * std::tanh(hl / t) / hl;
                    float e = std::min(1.0f, edge[i] * 1.5f);
                    g *= lim + (1.0f - lim) * e;
                }
            }
            for (int k = 0; k < 3; ++k) high[k][i] *= g;
        });
    }

    // 5. Recombine, with optional post sharpen.
    Plane out[3];
    for (int k = 0; k < 3; ++k) {
        out[k].resize(N);
        forEach(c, [&](std::size_t i) { out[k][i] = low[k][i] + high[k][i]; });
        if (p.postSharpen > 0.0f) {
            Plane soft = blurred(c, out[k], std::max(0.6f, 0.8f * scale), tmp);
            forEach(c, [&](std::size_t i) { out[k][i] += p.postSharpen * (out[k][i] - soft[i]); });
        }
    }

    // 6. Strength map: mask (softened, inverted) times mix.
    Plane m(N, std::clamp(p.mix, 0.0f, 1.0f));
    if (mask) {
        parallelFor(c.h, threads, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < c.w; ++x) {
                    std::size_t i = static_cast<std::size_t>(y) * c.w + x;
                    m[i] = std::clamp(mask[i], 0.0f, 1.0f);
                }
        });
        gauss(c, m.data(), p.maskSoftness * scale, tmp);
        const float mix = std::clamp(p.mix, 0.0f, 1.0f);
        forEach(c, [&](std::size_t i) {
            float v = std::clamp(m[i], 0.0f, 1.0f);
            m[i] = (p.invertMask ? 1.0f - v : v) * mix;
        });
    } else if (p.invertMask) {
        std::fill(m.begin(), m.end(), 0.0f);
    }
    if (p.skinKey) {
        // Soft qualifier on the hue of the (unsmoothed) colour layer, so the
        // key follows skin tone regions rather than pores and noise.
        const float hue0 = p.skinHue * 3.14159265f / 180.0f;
        const float width = std::max(1.0f, p.skinHueWidth) * 3.14159265f / 180.0f;
        const float minSat = std::max(1e-3f, p.skinMinSat);
        Plane key(N);
        forEach(c, [&](std::size_t i) {
            float r = fromWork(lowOrig[0][i], ws), g = fromWork(lowOrig[1][i], ws), b = fromWork(lowOrig[2][i], ws);
            float y = luma(r, g, b);
            float cb = b - y, cr = r - y;
            float chroma = std::sqrt(cb * cb + cr * cr);
            float sat = chroma / std::max(std::fabs(y), 0.02f);
            float d = std::remainder(std::atan2(-cb, cr) - hue0, 2.0f * 3.14159265f) / width;
            float hueW = std::exp(-d * d * 2.0f);
            float satW = std::clamp((sat - 0.5f * minSat) / minSat, 0.0f, 1.0f);
            key[i] = hueW * satW;
        });
        gauss(c, key.data(), std::max(1.0f, 2.0f * scale), tmp);
        forEach(c, [&](std::size_t i) { m[i] *= key[i]; });
    }

    // 7. Write the requested view.
    const int splitX = static_cast<int>(std::clamp(p.splitPosition, 0.0f, 1.0f) * c.w);
    const int lineW = std::max(1, static_cast<int>(scale + 0.5f));
    parallelFor(c.h, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const float* s = src.row(y);
            float* d = dst.row(y);
            std::size_t o = static_cast<std::size_t>(y) * c.w;
            for (int x = 0; x < c.w; ++x, s += src.channels, d += dst.channels) {
                std::size_t i = o + x;
                float v[3];
                switch (p.view) {
                case View::ColorLayer:
                    for (int k = 0; k < 3; ++k) v[k] = fromWork(low[k][i], ws);
                    break;
                case View::TextureLayer:
                    for (int k = 0; k < 3; ++k) v[k] = 0.5f + high[k][i];
                    break;
                case View::Mask:
                    v[0] = v[1] = v[2] = m[i];
                    break;
                case View::SplitScreen:
                    if (x >= splitX && x < splitX + lineW) {
                        v[0] = v[1] = v[2] = 1.0f;
                        break;
                    }
                    if (x < splitX) {
                        for (int k = 0; k < 3; ++k) v[k] = s[k];
                        break;
                    }
                    [[fallthrough]];
                case View::Result:
                default:
                    for (int k = 0; k < 3; ++k) v[k] = s[k] + (fromWork(out[k][i], ws) - s[k]) * m[i];
                    break;
                }
                for (int k = 0; k < 3; ++k) d[k] = v[k];
                if (dst.channels == 4) d[3] = src.channels == 4 ? s[3] : 1.0f;
            }
        }
    });
}

}  // namespace sep
