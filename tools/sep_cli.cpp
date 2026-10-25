// Command line front end for the frequency separation core.
// Usage: sep_cli in.pfm out.pfm [mask=mask.pfm] [separation=6] [colorSmooth=14] ...
// PFM (portable float map) keeps the test loop free of image library deps;
// tools/pfm.py converts to and from PNG.
#include "../src/core/FreqSep.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

struct Pfm {
    int w = 0, h = 0, c = 0;
    std::vector<float> px;
};

bool readPfm(const std::string& path, Pfm& img)
{
    std::ifstream f(path, std::ios::binary);
    std::string magic;
    float scale;
    if (!(f >> magic >> img.w >> img.h >> scale)) return false;
    f.get();
    img.c = magic == "PF" ? 3 : magic == "Pf" ? 1 : 0;
    if (!img.c || scale > 0) return false;  // only little endian
    img.px.resize(static_cast<size_t>(img.w) * img.h * img.c);
    f.read(reinterpret_cast<char*>(img.px.data()), img.px.size() * sizeof(float));
    return bool(f);
}

bool writePfm(const std::string& path, const Pfm& img)
{
    std::ofstream f(path, std::ios::binary);
    f << (img.c == 3 ? "PF" : "Pf") << "\n" << img.w << " " << img.h << "\n-1.0\n";
    f.write(reinterpret_cast<const char*>(img.px.data()), img.px.size() * sizeof(float));
    return bool(f);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "usage: sep_cli in.pfm out.pfm [key=value ...]\n";
        return 2;
    }
    Pfm in, mask;
    if (!readPfm(argv[1], in) || in.c != 3) {
        std::cerr << "cannot read RGB PFM " << argv[1] << "\n";
        return 1;
    }

    sep::Params p;
    std::map<std::string, float*> floats = {
        {"separation", &p.separation},   {"colorSmooth", &p.colorSmooth}, {"edgeProtect", &p.edgeProtect},
        {"colorAmount", &p.colorAmount}, {"midtoneDetail", &p.midtoneDetail}, {"texture", &p.texture},
        {"blemish", &p.blemish},         {"postSharpen", &p.postSharpen}, {"mix", &p.mix},
        {"maskSoftness", &p.maskSoftness}, {"split", &p.splitPosition}, {"skinHue", &p.skinHue},
        {"skinHueWidth", &p.skinHueWidth}, {"skinMinSat", &p.skinMinSat},
    };
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (eq == std::string::npos) continue;
        std::string k = a.substr(0, eq), v = a.substr(eq + 1);
        if (floats.count(k)) *floats[k] = std::strtof(v.c_str(), nullptr);
        else if (k == "view") p.view = static_cast<sep::View>(std::atoi(v.c_str()));
        else if (k == "linear") p.space = std::atoi(v.c_str()) ? sep::WorkingSpace::Linear : sep::WorkingSpace::AsIs;
        else if (k == "skinKey") p.skinKey = std::atoi(v.c_str()) != 0;
        else if (k == "invertMask") p.invertMask = std::atoi(v.c_str()) != 0;
        else if (k == "mask") {
            if (!readPfm(v, mask) || mask.w != in.w || mask.h != in.h) {
                std::cerr << "bad mask " << v << "\n";
                return 1;
            }
        } else std::cerr << "unknown parameter " << k << "\n";
    }

    std::vector<float> maskPlane;
    if (!mask.px.empty()) {
        maskPlane.resize(static_cast<size_t>(in.w) * in.h);
        for (size_t i = 0; i < maskPlane.size(); ++i) maskPlane[i] = mask.px[i * mask.c];
    }

    Pfm out = in;
    sep::ImageView s{in.px.data(), in.w, in.h, 3, static_cast<std::ptrdiff_t>(in.w) * 3};
    sep::ImageView d{out.px.data(), out.w, out.h, 3, static_cast<std::ptrdiff_t>(out.w) * 3};
    auto t0 = std::chrono::steady_clock::now();
    sep::process(s, d, maskPlane.empty() ? nullptr : maskPlane.data(), p);
    auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << in.w << "x" << in.h << " processed in " << ms << " ms\n";
    return writePfm(argv[2], out) ? 0 : 1;
}
