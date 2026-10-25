// Sanity checks for the frequency separation core on synthetic images.
#include "../src/core/FreqSep.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

struct Img {
    int w, h;
    std::vector<float> px;
    Img(int w_, int h_) : w(w_), h(h_), px(static_cast<size_t>(w_) * h_ * 4, 1.0f) {}
    sep::ImageView view() { return {px.data(), w, h, 4, static_cast<std::ptrdiff_t>(w) * 4}; }
    float* at(int x, int y) { return &px[(static_cast<size_t>(y) * w + x) * 4]; }
};

// Left half dark skin-ish tone, right half light, with blotchy low-frequency
// noise plus fine "pore" noise on top.
Img makeTest(int w, int h)
{
    Img img(w, h);
    std::mt19937 rng(1);
    std::normal_distribution<float> fine(0.0f, 0.01f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float base = x < w / 2 ? 0.25f : 0.75f;
            float blotch = 0.04f * std::sin(x * 0.15f) * std::sin(y * 0.18f);
            float* p = img.at(x, y);
            float n = fine(rng);
            p[0] = base + blotch + n;
            p[1] = 0.8f * (base + blotch) + n;
            p[2] = 0.6f * (base + blotch) + n;
            p[3] = 1.0f;
        }
    return img;
}

double regionStd(Img& img, int x0, int x1, int y0, int y1, int c)
{
    double s = 0, s2 = 0;
    int n = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            double v = img.at(x, y)[c];
            s += v;
            s2 += v * v;
            ++n;
        }
    double m = s / n;
    return std::sqrt(s2 / n - m * m);
}

}  // namespace

int main()
{
    const int W = 960, H = 540;
    Img src = makeTest(W, H);

    {
        Img out(W, H);
        sep::Params p;
        p.mix = 0.0f;
        sep::process(src.view(), out.view(), nullptr, p);
        float maxDiff = 0;
        for (size_t i = 0; i < src.px.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(src.px[i] - out.px[i]));
        check(maxDiff < 1e-6f, "mix 0 returns the source");
    }
    {
        Img out(W, H);
        sep::Params p;
        p.colorAmount = 0.0f;
        sep::process(src.view(), out.view(), nullptr, p);
        float maxDiff = 0;
        for (size_t i = 0; i < src.px.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(src.px[i] - out.px[i]));
        check(maxDiff < 1e-4f, "split + recombine without changes is lossless");
    }
    {
        Img out(W, H);
        sep::Params p;
        p.colorSmooth = 30.0f;
        p.edgeProtect = 0.2f;
        sep::process(src.view(), out.view(), nullptr, p);
        double before = regionStd(src, 20, W / 2 - 40, 20, H - 20, 0);
        double after = regionStd(out, 20, W / 2 - 40, 20, H - 20, 0);
        std::printf("      blotch std %.4f -> %.4f\n", before, after);
        check(after < before * 0.75, "colour layer smoothing reduces blotches");
        check(after > 0.008, "fine texture (pores) survives");
        float edgeL = out.at(W / 2 - 3, H / 2)[0], edgeR = out.at(W / 2 + 2, H / 2)[0];
        check(edgeR - edgeL > 0.4f, "strong edge is preserved");
        bool finite = true;
        for (float v : out.px) finite = finite && std::isfinite(v);
        check(finite, "output is finite");
    }
    {
        Img out(W, H);
        sep::Params p;
        std::vector<float> mask(static_cast<size_t>(W) * H, 0.0f);
        sep::process(src.view(), out.view(), mask.data(), p);
        float maxDiff = 0;
        for (size_t i = 0; i < src.px.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(src.px[i] - out.px[i]));
        check(maxDiff < 1e-6f, "empty mask leaves the image untouched");
    }
    {
        Img out(W, H);
        sep::Params p;
        p.space = sep::WorkingSpace::Linear;
        p.colorAmount = 0.0f;
        sep::process(src.view(), out.view(), nullptr, p);
        float maxDiff = 0;
        for (size_t i = 0; i < src.px.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(src.px[i] - out.px[i]));
        check(maxDiff < 1e-4f, "linear working space round-trips");
    }

    std::printf(failures ? "%d failure(s)\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
