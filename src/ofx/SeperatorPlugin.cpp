// OFX wrapper around the frequency separation core (src/core/FreqSep.h).
// Target host: DaVinci Resolve (macOS/Windows/Linux). Float RGBA/RGB only.
// On macOS the whole pipeline runs on the GPU when the host offers Metal.
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "ofxsImageEffect.h"

#include "../core/FreqSep.h"
#ifdef __APPLE__
#include "../metal/FreqSepMetal.h"
#endif

#define kPluginName "LineSeparator Skin"
#define kPluginGrouping "LineSeparator"
#define kPluginDescription \
    "Frequency separation for moving images. Splits the picture into a colour layer and a texture layer, " \
    "evens out the colour layer (blotches, redness, uneven tone) and keeps pores and fine detail."
#define kPluginIdentifier "dk.line21.seperator.skin"
#define kPluginVersionMajor 0
#define kPluginVersionMinor 1

#define kClipMask "Mask"

namespace {

int choiceAt(OFX::ChoiceParam* c, double t)
{
    int v = 0;
    c->getValueAtTime(t, v);
    return v;
}

enum MaskSource { eMaskNone = 0, eMaskInput, eMaskSourceAlpha };

class SeperatorPlugin : public OFX::ImageEffect {
public:
    explicit SeperatorPlugin(OfxImageEffectHandle handle)
        : ImageEffect(handle)
    {
        _dst = fetchClip(kOfxImageEffectOutputClipName);
        _src = fetchClip(kOfxImageEffectSimpleSourceClipName);
        _mask = fetchClip(kClipMask);

        _separation = fetchDoubleParam("separation");
        _colorSmooth = fetchDoubleParam("colorSmooth");
        _edgeProtect = fetchDoubleParam("edgeProtect");
        _colorAmount = fetchDoubleParam("colorAmount");
        _midtoneDetail = fetchDoubleParam("midtoneDetail");
        _texture = fetchDoubleParam("texture");
        _blemish = fetchDoubleParam("blemish");
        _postSharpen = fetchDoubleParam("postSharpen");
        _mix = fetchDoubleParam("mix");

        _maskSource = fetchChoiceParam("maskSource");
        _skinKey = fetchBooleanParam("skinKey");
        _skinHue = fetchDoubleParam("skinHue");
        _skinHueWidth = fetchDoubleParam("skinHueWidth");
        _skinMinSat = fetchDoubleParam("skinMinSat");
        _maskSoftness = fetchDoubleParam("maskSoftness");
        _invertMask = fetchBooleanParam("invertMask");

        _workingSpace = fetchChoiceParam("workingSpace");
        _view = fetchChoiceParam("view");
        _split = fetchDoubleParam("splitPosition");
    }

    void render(const OFX::RenderArguments& args) override;
    bool isIdentity(const OFX::IsIdentityArguments& args, OFX::Clip*& identityClip, double& identityTime) override;

private:
    sep::Params paramsAt(double t) const;
#ifdef __APPLE__
    void renderMetal(const OFX::RenderArguments& args, OFX::Image& src, OFX::Image& dst, int srcCh, int dstCh,
                     int maskSource, const sep::Params& p);
#endif

    OFX::Clip *_dst, *_src, *_mask;
    OFX::DoubleParam *_separation, *_colorSmooth, *_edgeProtect, *_colorAmount, *_midtoneDetail, *_texture,
        *_blemish, *_postSharpen, *_mix, *_skinHue, *_skinHueWidth, *_skinMinSat, *_maskSoftness, *_split;
    OFX::ChoiceParam *_maskSource, *_workingSpace, *_view;
    OFX::BooleanParam *_skinKey, *_invertMask;
};

sep::Params SeperatorPlugin::paramsAt(double t) const
{
    sep::Params p;
    p.separation = static_cast<float>(_separation->getValueAtTime(t));
    p.colorSmooth = static_cast<float>(_colorSmooth->getValueAtTime(t));
    p.edgeProtect = static_cast<float>(_edgeProtect->getValueAtTime(t));
    p.colorAmount = static_cast<float>(_colorAmount->getValueAtTime(t));
    p.midtoneDetail = static_cast<float>(_midtoneDetail->getValueAtTime(t));
    p.texture = static_cast<float>(_texture->getValueAtTime(t));
    p.blemish = static_cast<float>(_blemish->getValueAtTime(t));
    p.postSharpen = static_cast<float>(_postSharpen->getValueAtTime(t));
    p.mix = static_cast<float>(_mix->getValueAtTime(t));
    p.skinKey = _skinKey->getValueAtTime(t);
    p.skinHue = static_cast<float>(_skinHue->getValueAtTime(t));
    p.skinHueWidth = static_cast<float>(_skinHueWidth->getValueAtTime(t));
    p.skinMinSat = static_cast<float>(_skinMinSat->getValueAtTime(t));
    p.maskSoftness = static_cast<float>(_maskSoftness->getValueAtTime(t));
    p.invertMask = _invertMask->getValueAtTime(t);
    p.space = static_cast<sep::WorkingSpace>(choiceAt(_workingSpace, t));
    p.view = static_cast<sep::View>(choiceAt(_view, t));
    p.splitPosition = static_cast<float>(_split->getValueAtTime(t));
    return p;
}

bool SeperatorPlugin::isIdentity(const OFX::IsIdentityArguments& args, OFX::Clip*& identityClip, double& identityTime)
{
    if (choiceAt(_view, args.time) == static_cast<int>(sep::View::Result) && _mix->getValueAtTime(args.time) <= 0.0) {
        identityClip = _src;
        identityTime = args.time;
        return true;
    }
    return false;
}

int channelsOf(OFX::PixelComponentEnum c)
{
    switch (c) {
    case OFX::ePixelComponentRGBA: return 4;
    case OFX::ePixelComponentRGB: return 3;
    case OFX::ePixelComponentAlpha: return 1;
    default: return 0;
    }
}

void SeperatorPlugin::render(const OFX::RenderArguments& args)
{
    std::unique_ptr<OFX::Image> dst(_dst->fetchImage(args.time));
    std::unique_ptr<OFX::Image> src(_src->fetchImage(args.time));
    if (!dst || !src) OFX::throwSuiteStatusException(kOfxStatFailed);
    if (dst->getPixelDepth() != OFX::eBitDepthFloat || src->getPixelDepth() != OFX::eBitDepthFloat)
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);

    const int srcCh = channelsOf(src->getPixelComponents());
    const int dstCh = channelsOf(dst->getPixelComponents());
    if (srcCh < 3 || dstCh < 3) OFX::throwSuiteStatusException(kOfxStatErrUnsupported);

    const OfxRectI sb = src->getBounds();
    const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;
    if (w <= 0 || h <= 0) return;

    sep::Params p = paramsAt(args.time);
    const int maskSource = choiceAt(_maskSource, args.time);

#ifdef __APPLE__
    if (args.isEnabledMetalRender) {
        renderMetal(args, *src, *dst, srcCh, dstCh, maskSource, p);
        return;
    }
#endif

    // Build the external mask plane (row 0 = bottom row of src, like the image).
    std::vector<float> mask;
    if (maskSource == eMaskSourceAlpha && srcCh == 4) {
        mask.resize(static_cast<size_t>(w) * h);
        for (int y = 0; y < h; ++y) {
            const float* s = static_cast<const float*>(src->getPixelAddress(sb.x1, sb.y1 + y));
            for (int x = 0; x < w; ++x) mask[static_cast<size_t>(y) * w + x] = s[x * 4 + 3];
        }
    } else if (maskSource == eMaskInput && _mask && _mask->isConnected()) {
        std::unique_ptr<OFX::Image> m(_mask->fetchImage(args.time));
        if (m && m->getPixelDepth() == OFX::eBitDepthFloat) {
            const int mc = channelsOf(m->getPixelComponents());
            const int off = mc == 4 ? 3 : 0;
            mask.assign(static_cast<size_t>(w) * h, 0.0f);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const float* mp = static_cast<const float*>(m->getPixelAddress(sb.x1 + x, sb.y1 + y));
                    if (mp) mask[static_cast<size_t>(y) * w + x] = mp[off];
                }
        }
    }

    // Process the whole source into a scratch buffer (the filters need context
    // beyond the render window), then copy the render window out.
    std::vector<float> out(static_cast<size_t>(w) * h * dstCh);
    sep::ImageView sv{static_cast<float*>(src->getPixelAddress(sb.x1, sb.y1)), w, h, srcCh,
                      static_cast<std::ptrdiff_t>(src->getRowBytes() / static_cast<int>(sizeof(float)))};
    sep::ImageView dv{out.data(), w, h, dstCh, static_cast<std::ptrdiff_t>(w) * dstCh};
    sep::process(sv, dv, mask.empty() ? nullptr : mask.data(), p);

    const OfxRectI db = dst->getBounds();
    const int x1 = std::max({args.renderWindow.x1, db.x1, sb.x1});
    const int x2 = std::min({args.renderWindow.x2, db.x2, sb.x2});
    const int y1 = std::max({args.renderWindow.y1, db.y1, sb.y1});
    const int y2 = std::min({args.renderWindow.y2, db.y2, sb.y2});
    if (x1 >= x2) return;
    for (int y = y1; y < y2; ++y) {
        if (abort()) return;
        float* d = static_cast<float*>(dst->getPixelAddress(x1, y));
        const float* o = out.data() + (static_cast<size_t>(y - sb.y1) * w + (x1 - sb.x1)) * dstCh;
        std::memcpy(d, o, sizeof(float) * dstCh * (x2 - x1));
    }
}

#ifdef __APPLE__
// Images are MTLBuffers here; getPixelData() returns the buffer handle.
void SeperatorPlugin::renderMetal(const OFX::RenderArguments& args, OFX::Image& src, OFX::Image& dst, int srcCh,
                                  int dstCh, int maskSource, const sep::Params& p)
{
    const OfxRectI sb = src.getBounds(), db = dst.getBounds();
    if (sb.x1 != db.x1 || sb.y1 != db.y1 || sb.x2 != db.x2 || sb.y2 != db.y2)
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    const int w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;

    sep::MetalImage s{src.getPixelData(), w, h, srcCh, src.getRowBytes() / static_cast<int>(sizeof(float))};
    sep::MetalImage d{dst.getPixelData(), w, h, dstCh, dst.getRowBytes() / static_cast<int>(sizeof(float))};

    sep::MetalMask mask;
    std::unique_ptr<OFX::Image> m;
    if (maskSource == eMaskSourceAlpha && srcCh == 4) {
        mask = {s.buffer, 4, 3, s.rowStride, 0, 0, w, h};
    } else if (maskSource == eMaskInput && _mask && _mask->isConnected()) {
        m.reset(_mask->fetchImage(args.time));
        if (m && m->getPixelDepth() == OFX::eBitDepthFloat) {
            const OfxRectI mb = m->getBounds();
            const int mc = channelsOf(m->getPixelComponents());
            mask = {m->getPixelData(), mc, mc == 4 ? 3 : 0, m->getRowBytes() / static_cast<int>(sizeof(float)),
                    sb.x1 - mb.x1, sb.y1 - mb.y1, mb.x2 - mb.x1, mb.y2 - mb.y1};
        }
    }

    std::string err;
    if (!sep::processMetal(args.pMetalCmdQ, s, d, mask.buffer ? &mask : nullptr, p, false, &err)) {
        setPersistentMessage(OFX::Message::eMessageError, "", err);
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }
}
#endif

mDeclarePluginFactory(SeperatorPluginFactory, {}, {});

OFX::DoubleParamDescriptor* defineDouble(OFX::ImageEffectDescriptor& desc, const char* name, const char* label,
                                         const char* hint, double def, double min, double max,
                                         OFX::GroupParamDescriptor* group, OFX::PageParamDescriptor* page)
{
    OFX::DoubleParamDescriptor* d = desc.defineDoubleParam(name);
    d->setLabels(label, label, label);
    d->setHint(hint);
    d->setDefault(def);
    d->setRange(min, max);
    d->setDisplayRange(min, max);
    d->setIncrement((max - min) / 200.0);
    d->setDigits(2);
    d->setDoubleType(OFX::eDoubleTypePlain);
    d->setAnimates(true);
    if (group) d->setParent(*group);
    if (page) page->addChild(*d);
    return d;
}

void SeperatorPluginFactory::describe(OFX::ImageEffectDescriptor& desc)
{
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);

    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);  // the core spreads work over all cores itself
    desc.setSupportsMultiResolution(true);
    desc.setSupportsTiles(false);       // blurs need the whole frame
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
#ifdef __APPLE__
    desc.setSupportsMetalRender(true);
#endif
}

void SeperatorPluginFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum /*context*/)
{
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);
    src->setIsMask(false);

    OFX::ClipDescriptor* mask = desc.defineClip(kClipMask);
    mask->addSupportedComponent(OFX::ePixelComponentAlpha);
    mask->addSupportedComponent(OFX::ePixelComponentRGBA);
    mask->setTemporalClipAccess(false);
    mask->setOptional(true);
    mask->setSupportsTiles(false);
    mask->setIsMask(true);

    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentRGB);
    dst->setSupportsTiles(false);

    OFX::PageParamDescriptor* page = desc.definePageParam("Controls");

    OFX::GroupParamDescriptor* fs = desc.defineGroupParam("grpSeparation");
    fs->setLabels("Frequency Separation", "Frequency Separation", "Frequency Separation");
    page->addChild(*fs);
    defineDouble(desc, "separation", "Separating Filter",
                 "Size of the split between colour and texture layers (pixels at 1080p). "
                 "Larger moves more detail into the texture layer.",
                 6.0, 0.5, 40.0, fs, page);
    defineDouble(desc, "colorSmooth", "Color Layer Filter",
                 "How much the colour layer is evened out (pixels at 1080p).", 14.0, 0.0, 80.0, fs, page);
    defineDouble(desc, "edgeProtect", "Edge Protection",
                 "Keeps strong edges (eyes, lips, jaw) out of the colour smoothing.", 0.5, 0.0, 1.0, fs, page);
    defineDouble(desc, "colorAmount", "Color Layer Amount", "Blend of the smoothed colour layer.", 1.0, 0.0, 1.0,
                 fs, page);
    defineDouble(desc, "midtoneDetail", "Midtone Detail",
                 "Contrast of the mid frequencies just below the texture. Negative flattens bumps.", 0.0, -1.0, 1.0,
                 fs, page);
    defineDouble(desc, "texture", "Texture", "Gain of the texture layer (1 = untouched).", 1.0, 0.0, 2.0, fs, page);
    defineDouble(desc, "blemish", "Blemish Reduction",
                 "Softens dark spots in the texture while keeping pores and edges.", 0.0, 0.0, 1.0, fs, page);
    defineDouble(desc, "postSharpen", "Post-Sharpen", "Sharpens the finest detail after recombination.", 0.0, 0.0,
                 2.0, fs, page);
    defineDouble(desc, "mix", "Mix", "Overall strength.", 1.0, 0.0, 1.0, fs, page);

    OFX::GroupParamDescriptor* mg = desc.defineGroupParam("grpMask");
    mg->setLabels("Masking", "Masking", "Masking");
    page->addChild(*mg);
    {
        OFX::ChoiceParamDescriptor* c = desc.defineChoiceParam("maskSource");
        c->setLabels("Mask Source", "Mask Source", "Mask Source");
        c->setHint("Where the effect is applied. Resolve users can also use the node's qualifier/windows as usual.");
        c->appendOption("Whole Frame");
        c->appendOption("Mask Input");
        c->appendOption("Source Alpha");
        c->setDefault(eMaskNone);
        c->setParent(*mg);
        page->addChild(*c);
    }
    {
        OFX::BooleanParamDescriptor* b = desc.defineBooleanParam("skinKey");
        b->setLabels("Skin Key", "Skin Key", "Skin Key");
        b->setHint("Limit the effect to skin tones (multiplies with the mask source).");
        b->setDefault(true);
        b->setParent(*mg);
        page->addChild(*b);
    }
    defineDouble(desc, "skinHue", "Skin Hue", "Centre hue of the skin key, degrees.", 55.0, 0.0, 120.0, mg, page);
    defineDouble(desc, "skinHueWidth", "Skin Hue Width", "Tolerance either side of the skin hue, degrees.", 25.0,
                 2.0, 90.0, mg, page);
    defineDouble(desc, "skinMinSat", "Skin Min Saturation", "Greys and whites below this fade out of the key.",
                 0.08, 0.0, 0.5, mg, page);
    defineDouble(desc, "maskSoftness", "Soften Mask", "Blur of the final mask (pixels at 1080p).", 4.0, 0.0, 50.0,
                 mg, page);
    {
        OFX::BooleanParamDescriptor* b = desc.defineBooleanParam("invertMask");
        b->setLabels("Invert Mask", "Invert Mask", "Invert Mask");
        b->setDefault(false);
        b->setParent(*mg);
        page->addChild(*b);
    }

    OFX::GroupParamDescriptor* og = desc.defineGroupParam("grpOutput");
    og->setLabels("Color Management & View", "Color Management & View", "Color Management & View");
    page->addChild(*og);
    {
        OFX::ChoiceParamDescriptor* c = desc.defineChoiceParam("workingSpace");
        c->setLabels("Input Is", "Input Is", "Input Is");
        c->setHint("Pick Scene Linear when the timeline/node is linear, so the separation happens in a log-like space.");
        c->appendOption("Gamma / Log (as is)");
        c->appendOption("Scene Linear");
        c->setDefault(0);
        c->setParent(*og);
        page->addChild(*c);
    }
    {
        OFX::ChoiceParamDescriptor* c = desc.defineChoiceParam("view");
        c->setLabels("View", "View", "View");
        c->appendOption("Result");
        c->appendOption("Color Layer");
        c->appendOption("Texture Layer");
        c->appendOption("Mask");
        c->appendOption("Split Before/After");
        c->setDefault(0);
        c->setParent(*og);
        page->addChild(*c);
    }
    defineDouble(desc, "splitPosition", "Split Position", "Where the before/after split sits.", 0.5, 0.0, 1.0, og,
                 page);
}

OFX::ImageEffect* SeperatorPluginFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum /*context*/)
{
    return new SeperatorPlugin(handle);
}

}  // namespace

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids)
{
    static SeperatorPluginFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&p);
}
}  // namespace Plugin
}  // namespace OFX
