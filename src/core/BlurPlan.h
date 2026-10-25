// Blur sizing shared by the CPU core and the Metal path, so both compute the
// same thing.
#pragma once

#include <algorithm>
#include <cmath>

namespace sep {

// Radii of three box filters approximating a gaussian of the given sigma.
inline void boxRadiiForGauss(float sigma, int radii[3])
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

// Integer factor a full resolution blur of this sigma runs at.
inline int smoothFactor(float sigma) { return std::clamp(static_cast<int>(sigma / 3.0f), 1, 8); }

// Factor the guided filter coefficients are solved at.
inline int guidedFactor(float sigmaG, float sigmaSep)
{
    return std::clamp(static_cast<int>(std::min(sigmaG / 3.0f, sigmaSep)), 1, 8);
}

// Shared parameter mapping.
inline float sigmaSeparation(float separation, float scale) { return std::max(0.5f, separation * scale); }
inline float sigmaGuided(float colorSmooth, float scale) { return std::max(0.5f, colorSmooth * scale * 0.5f); }
inline float guidedEps(float edgeProtect) { return std::pow(10.0f, -2.0f - 3.0f * std::clamp(edgeProtect, 0.0f, 1.0f)); }
inline float blemishThreshold(float blemish) { return 0.12f * (1.0f - blemish) * (1.0f - blemish) + 0.012f; }
inline float sharpenSigma(float scale) { return std::max(0.6f, 0.8f * scale); }
inline float skinKeySigma(float scale) { return std::max(1.0f, 2.0f * scale); }

}  // namespace sep
