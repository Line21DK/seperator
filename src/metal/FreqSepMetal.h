// Metal implementation of sep::process for hosts that hand us MTLBuffers
// (DaVinci Resolve on macOS). Plain C++ interface; Objective-C stays in the .mm.
#pragma once

#include <string>
#include <vector>

#include "../core/FreqSep.h"

namespace sep {

struct MetalImage {
    void* buffer = nullptr;  // id<MTLBuffer>, float pixels, origin = first row
    int width = 0;
    int height = 0;
    int channels = 4;
    int rowStride = 0;  // in floats
};

struct MetalMask {
    void* buffer = nullptr;  // id<MTLBuffer>
    int channels = 1;
    int channelOffset = 0;
    int rowStride = 0;  // in floats
    int dx = 0, dy = 0;  // mask pixel = source pixel + (dx, dy)
    int width = 0, height = 0;
    // Alternatively a mask in CPU memory: width x height floats, one channel,
    // row 0 = first row of the source. Copied to the GPU per frame.
    const float* host = nullptr;
};

// Encodes the whole pipeline on the host's command queue and commits it.
// When wait is true, blocks until the GPU is done (tests). Returns false and
// fills err on failure.
bool processMetal(void* commandQueue, const MetalImage& src, const MetalImage& dst, const MetalMask* mask,
                  const Params& p, bool wait, std::string* err);

// Box-filtered RGB thumbnail of src (interleaved, row 0 = first row), read
// back to the CPU. Waits for the GPU. Used for the person mask.
bool thumbnailMetal(void* commandQueue, const MetalImage& src, int factor, std::vector<float>& rgb, int& width,
                    int& height, std::string* err);

}  // namespace sep
