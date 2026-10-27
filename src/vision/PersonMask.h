// AI person mask on macOS via Apple's Vision framework (person segmentation,
// macOS 12+, runs on the Neural Engine/GPU). Plain C++ interface; the
// Objective-C stays in the .mm.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../core/FreqSep.h"

namespace sep {

enum class PersonQuality { Fast = 0, Balanced, Accurate };

// Reduced-resolution RGB copy of the frame that the segmentation runs on.
// Interleaved RGB floats, row 0 = bottom row (like the source image).
struct Thumb {
    std::vector<float> rgb;
    int width = 0, height = 0;
};

// Box-filter factor that brings `height` down to what the quality level needs.
int thumbFactor(int height, PersonQuality q);

// CPU thumbnail of src (the Metal path makes the same thing on the GPU).
void makeThumb(const ImageView& src, int factor, Thumb& out);

// Hash of the thumbnail, so an unchanged frame can reuse its mask.
uint64_t thumbHash(const Thumb& t);

// True when the OS has person segmentation (macOS 12 or newer).
bool personMaskAvailable();

// Segments the people in `t` and writes a width x height mask (0..1, row 0 =
// bottom) into `mask`. `space` says how to encode the pixels for the model.
bool personMask(const Thumb& t, WorkingSpace space, PersonQuality q, int width, int height,
                std::vector<float>& mask, std::string* err);

}  // namespace sep
