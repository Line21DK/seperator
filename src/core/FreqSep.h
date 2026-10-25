// Seperator: frequency separation core.
//
// Splits an image into a low-frequency "colour layer" and a high-frequency
// "texture layer", smooths the colour layer with an edge-preserving filter,
// optionally tames large texture spikes (blemishes) and recombines.
// Host-agnostic: the OFX plugin and the command line tool both call process().
#pragma once

#include <cstddef>
#include <vector>

namespace sep {

enum class View {
    Result = 0,
    ColorLayer,
    TextureLayer,
    Mask,
    SplitScreen,
};

enum class WorkingSpace {
    AsIs = 0,  // data is already perceptual (Rec.709 gamma, log, ...)
    Linear,    // scene linear; separation is done in a log-like domain
};

struct Params {
    // Filter sizes are in pixels at 1080 lines and scaled with resolution,
    // so a look is the same in a 1080p timeline and a 4K render.
    float separation = 6.0f;      // sigma of the split between colour and texture
    float colorSmooth = 14.0f;    // radius of the colour layer smoothing
    float edgeProtect = 0.5f;     // 0..1, higher keeps more edges in the colour layer
    float colorAmount = 1.0f;     // 0..1 blend of the smoothed colour layer
    float midtoneDetail = 0.0f;   // -1..1, mid-frequency contrast inside the colour layer
    float texture = 1.0f;         // 0..2 gain of the texture layer
    float blemish = 0.0f;         // 0..1 soft-limits dark texture spikes (spots, pimples)
    float postSharpen = 0.0f;     // 0..2 boosts the finest texture after recombination
    float mix = 1.0f;             // 0..1 overall strength

    // Masking. The external mask (if any) is multiplied by the skin key (if on).
    bool skinKey = false;
    float skinHue = 55.0f;        // degrees, angle of (Cr, -Cb); typical skin is 45..65
    float skinHueWidth = 25.0f;   // degrees of tolerance either side
    float skinMinSat = 0.08f;     // chroma/luma below this fades out (greys, whites)
    float maskSoftness = 0.0f;    // sigma of the mask blur, pixels at 1080 lines
    bool invertMask = false;

    WorkingSpace space = WorkingSpace::AsIs;
    View view = View::Result;
    float splitPosition = 0.5f;   // 0..1, used by View::SplitScreen
};

// Interleaved float image with 3 or 4 channels. Alpha (if any) passes through.
struct ImageView {
    float* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 4;
    std::ptrdiff_t rowStride = 0;  // in floats, may be negative (bottom-up hosts)

    float* row(int y) const { return data + rowStride * y; }
};

// Processes src into dst (same size, may not alias). mask is optional
// (width*height floats in 0..1, row-major, top or bottom up like src);
// nullptr means "everywhere".
void process(const ImageView& src, const ImageView& dst, const float* mask,
             const Params& p, int threads = 0);

}  // namespace sep
