
/**
 *  SangNom - VapourSynth Single Field Deinterlacer
 *
 *  Copyright (c) 2016 james1201
 *  Copyright (c) 2013 Victor Efimov
 *
 *  This project is licensed under the MIT license. Binaries are GPL v2.
 **/

#include <cmath>
#include <string>

#include "SangNom2.h"

static constexpr size_t alignment { 32 };

enum class SangNomOrderType
{
    SNOT_DFR = 0,   // double frame rate, user must call std.SeparateFields().std.DoubleWeave() before use this
    SNOT_SFR_KT,    // single frame rate, keep top field
    SNOT_SFR_KB     // single frame rate, keep bottom field
};

template <typename T, typename IType>
static AVS_FORCEINLINE IType loadPixel(const T* srcp, int curPos, int offset, int width) noexcept
{
    int reqPos{ curPos + offset };
    if (reqPos >= 0 && reqPos < width)
        return srcp[reqPos];
    if (reqPos >= 0)
        return srcp[width - 1];
    return srcp[0];
}

template <typename T>
static AVS_FORCEINLINE T absDiff(const T& a, const T& b) noexcept
{
    return std::abs(a - b);
}

template <>
AVS_FORCEINLINE float absDiff<float>(const float& a, const float& b) noexcept
{
    return std::fabs(a - b);
}

template <typename T>
static AVS_FORCEINLINE T avg(const T& a, const T& b) noexcept
{
    return (a + b + 1) >> 1;
}

template <>
AVS_FORCEINLINE float avg<float>(const float& a, const float& b) noexcept
{
    return (a + b) * static_cast<float>(1.0 / 2.0);
}

template <typename T, typename IType>
static AVS_FORCEINLINE IType calculateSangNom(const T& p1, const T& p2, const T& p3) noexcept
{
    IType sum{ static_cast<IType>(p1 * 4 + p2 * 5 - p3) };
    return static_cast<T>(sum >> 3);
}

template <>
AVS_FORCEINLINE float calculateSangNom(const float& p1, const float& p2, const float& p3) noexcept
{
    float sum{ p1 * 4 + p2 * 5 - p3 };
    return sum * static_cast<float>(1.0 / 8.0);
}

template <typename T, typename IType>
static AVS_FORCEINLINE void prepareBuffers_c(const T* srcp, const int srcStride, const int w, const int h, const int bufferStride, T* buffers[TOTAL_BUFFERS]) noexcept
{
    auto srcpn2{ srcp + srcStride * static_cast<int64_t>(2) };

    int bufferOffset{ bufferStride };

    for (int y{ 0 }; y < h / 2 - 1; ++y)
    {

        for (int x{ 0 }; x < w; ++x)
        {

            const IType currLineM3{ loadPixel<T, IType>(srcp, x, -3, w) };
            const IType currLineM2{ loadPixel<T, IType>(srcp, x, -2, w) };
            const IType currLineM1{ loadPixel<T, IType>(srcp, x, -1, w) };
            const IType currLine{ srcp[x] };
            const IType currLineP1{ loadPixel<T, IType>(srcp, x, 1, w) };
            const IType currLineP2{ loadPixel<T, IType>(srcp, x, 2, w) };
            const IType currLineP3{ loadPixel<T, IType>(srcp, x, 3, w) };

            const IType nextLineM3{ loadPixel<T, IType>(srcpn2, x, -3, w) };
            const IType nextLineM2{ loadPixel<T, IType>(srcpn2, x, -2, w) };
            const IType nextLineM1{ loadPixel<T, IType>(srcpn2, x, -1, w) };
            const IType nextLine{ srcpn2[x] };
            const IType nextLineP1{ loadPixel<T, IType>(srcpn2, x, 1, w) };
            const IType nextLineP2{ loadPixel<T, IType>(srcpn2, x, 2, w) };
            const IType nextLineP3{ loadPixel<T, IType>(srcpn2, x, 3, w) };

            const IType forwardSangNom1{ calculateSangNom<T, IType>(currLineM1, currLine, currLineP1) };
            const IType forwardSangNom2{ calculateSangNom<T, IType>(nextLineP1, nextLine, nextLineM1) };
            const IType backwardSangNom1{ calculateSangNom<T, IType>(currLineP1, currLine, currLineM1) };
            const IType backwardSangNom2{ calculateSangNom<T, IType>(nextLineM1, nextLine, nextLineP1) };

            buffers[as_int(Buffers::ADIFF_M3_P3)][bufferOffset + x] = static_cast<T>(absDiff(currLineM3, nextLineP3));
            buffers[as_int(Buffers::ADIFF_M2_P2)][bufferOffset + x] = static_cast<T>(absDiff(currLineM2, nextLineP2));
            buffers[as_int(Buffers::ADIFF_M1_P1)][bufferOffset + x] = static_cast<T>(absDiff(currLineM1, nextLineP1));
            buffers[as_int(Buffers::ADIFF_P0_M0)][bufferOffset + x] = static_cast<T>(absDiff(currLine, nextLine));
            buffers[as_int(Buffers::ADIFF_P1_M1)][bufferOffset + x] = static_cast<T>(absDiff(currLineP1, nextLineM1));
            buffers[as_int(Buffers::ADIFF_P2_M2)][bufferOffset + x] = static_cast<T>(absDiff(currLineP2, nextLineM2));
            buffers[as_int(Buffers::ADIFF_P3_M3)][bufferOffset + x] = static_cast<T>(absDiff(currLineP3, nextLineM3));

            buffers[as_int(Buffers::SG_FORWARD)][bufferOffset + x] = static_cast<T>(absDiff(forwardSangNom1, forwardSangNom2));
            buffers[as_int(Buffers::SG_REVERSE)][bufferOffset + x] = static_cast<T>(absDiff(backwardSangNom1, backwardSangNom2));
        }

        srcp += srcStride * static_cast<int64_t>(2);
        srcpn2 += static_cast<int64_t>(srcStride) * 2;
        bufferOffset += bufferStride;
    }
}

template <typename T, typename IType>
static AVS_FORCEINLINE void processBuffers_c(T* bufferp, IType* bufferLine, const int bufferStride, const int bufferHeight) noexcept
{
    auto bufferpc{ bufferp + bufferStride };
    auto bufferpp1{ bufferpc - bufferStride };
    auto bufferpn1{ bufferpc + bufferStride };

    for (int y{ 0 }; y < bufferHeight - 1; ++y)
    {

        for (int x{ 0 }; x < bufferStride; ++x)
        {
            bufferLine[x] = bufferpp1[x] + bufferpc[x] + bufferpn1[x];
        }

        for (int x{ 0 }; x < bufferStride; ++x)
        {

            const IType currLineM3{ loadPixel<IType, IType>(bufferLine, x, -3, bufferStride) };
            const IType currLineM2{ loadPixel<IType, IType>(bufferLine, x, -2, bufferStride) };
            const IType currLineM1{ loadPixel<IType, IType>(bufferLine, x, -1, bufferStride) };
            const IType currLine{ bufferLine[x] };
            const IType currLineP1{ loadPixel<IType, IType>(bufferLine, x, 1, bufferStride) };
            const IType currLineP2{ loadPixel<IType, IType>(bufferLine, x, 2, bufferStride) };
            const IType currLineP3{ loadPixel<IType, IType>(bufferLine, x, 3, bufferStride) };

            bufferpc[x] = (currLineM3 + currLineM2 + currLineM1 + currLine + currLineP1 + currLineP2 + currLineP3) / 16;
        }

        bufferpc += bufferStride;
        bufferpp1 += bufferStride;
        bufferpn1 += bufferStride;
    }
}

template <typename T, typename IType>
static AVS_FORCEINLINE void finalizePlane_c(T* dstp, const int dstStride, const int w, const int h, const int bufferStride, const T aaf, T* buffers[TOTAL_BUFFERS]) noexcept
{
    auto srcp{ dstp };
    auto srcpn2{ srcp + dstStride * static_cast<int64_t>(2) };
    auto dstpn{ dstp + dstStride };

    int bufferOffset{ bufferStride };

    for (int y{ 0 }; y < h / 2 - 1; ++y)
    {

        for (int x{ 0 }; x < w; ++x)
        {

            const IType currLineM3{ loadPixel<T, IType>(srcp, x, -3, w) };
            const IType currLineM2{ loadPixel<T, IType>(srcp, x, -2, w) };
            const IType currLineM1{ loadPixel<T, IType>(srcp, x, -1, w) };
            const IType currLine{ srcp[x] };
            const IType currLineP1{ loadPixel<T, IType>(srcp, x, 1, w) };
            const IType currLineP2{ loadPixel<T, IType>(srcp, x, 2, w) };
            const IType currLineP3{ loadPixel<T, IType>(srcp, x, 3, w) };

            const IType nextLineM3{ loadPixel<T, IType>(srcpn2, x, -3, w) };
            const IType nextLineM2{ loadPixel<T, IType>(srcpn2, x, -2, w) };
            const IType nextLineM1{ loadPixel<T, IType>(srcpn2, x, -1, w) };
            const IType nextLine{ srcpn2[x] };
            const IType nextLineP1{ loadPixel<T, IType>(srcpn2, x, 1, w) };
            const IType nextLineP2{ loadPixel<T, IType>(srcpn2, x, 2, w) };
            const IType nextLineP3{ loadPixel<T, IType>(srcpn2, x, 3, w) };

            const IType forwardSangNom1{ calculateSangNom<T, IType>(currLineM1, currLine, currLineP1) };
            const IType forwardSangNom2{ calculateSangNom<T, IType>(nextLineP1, nextLine, nextLineM1) };
            const IType backwardSangNom1{ calculateSangNom<T, IType>(currLineP1, currLine, currLineM1) };
            const IType backwardSangNom2{ calculateSangNom<T, IType>(nextLineM1, nextLine, nextLineP1) };

            IType buf[9];
            buf[0] = buffers[as_int(Buffers::ADIFF_M3_P3)][bufferOffset + x];
            buf[1] = buffers[as_int(Buffers::ADIFF_M2_P2)][bufferOffset + x];
            buf[2] = buffers[as_int(Buffers::ADIFF_M1_P1)][bufferOffset + x];
            buf[3] = buffers[as_int(Buffers::SG_FORWARD)][bufferOffset + x];
            buf[4] = buffers[as_int(Buffers::ADIFF_P0_M0)][bufferOffset + x];
            buf[5] = buffers[as_int(Buffers::SG_REVERSE)][bufferOffset + x];
            buf[6] = buffers[as_int(Buffers::ADIFF_P1_M1)][bufferOffset + x];
            buf[7] = buffers[as_int(Buffers::ADIFF_P2_M2)][bufferOffset + x];
            buf[8] = buffers[as_int(Buffers::ADIFF_P3_M3)][bufferOffset + x];

            IType minBuf{ buf[0] };
            for (int i{ 1 }; i < 9; ++i)
                minBuf = std::min(minBuf, buf[i]);

            ///////////////////////////////////////////////////////////////////////
            // the order of following code is important, don't change them
            if (buf[4] == minBuf || minBuf > aaf)
            {
                dstpn[x] = avg(currLine, nextLine);
            }
            else if (buf[5] == minBuf)
            {
                dstpn[x] = avg(backwardSangNom1, backwardSangNom2);
            }
            else if (buf[3] == minBuf)
            {
                dstpn[x] = avg(forwardSangNom1, forwardSangNom2);
            }
            else if (buf[6] == minBuf)
            {
                dstpn[x] = avg(currLineP1, nextLineM1);
            }
            else if (buf[2] == minBuf)
            {
                dstpn[x] = avg(currLineM1, nextLineP1);
            }
            else if (buf[7] == minBuf)
            {
                dstpn[x] = avg(currLineP2, nextLineM2);
            }
            else if (buf[1] == minBuf)
            {
                dstpn[x] = avg(currLineM2, nextLineP2);
            }
            else if (buf[8] == minBuf)
            {
                dstpn[x] = avg(currLineP3, nextLineM3);
            }
            else if (buf[0] == minBuf)
            {
                dstpn[x] = avg(currLineM3, nextLineP3);
            }
        }

        srcp += dstStride * static_cast<int64_t>(2);
        srcpn2 += dstStride * static_cast<int64_t>(2);
        dstpn += dstStride * static_cast<int64_t>(2);
        bufferOffset += bufferStride;
    }
}

template <typename T, typename IType>
void SangNom2::sangnom_c(uint8_t* __restrict dstp_, const int dstStride, const int w, const int h, int offset, int plane) noexcept
{
    T* __restrict dstp{ reinterpret_cast<T*>(dstp_) };
    T** buffers = reinterpret_cast<T**>(buffers_);
    IType* bufferLine{ reinterpret_cast<IType*>(bufferLine_.get()) };
    const size_t buffer_stride{ bufferStride / sizeof(T) };

    prepareBuffers_c<T, IType>(dstp + static_cast<int64_t>(offset) * dstStride, dstStride, w, h, buffer_stride, buffers);

    for (int i{ 0 }; i < TOTAL_BUFFERS; ++i)
        processBuffers_c<T, IType>(buffers[i], bufferLine, buffer_stride, bufferHeight);

    finalizePlane_c<T, IType>(dstp + static_cast<int64_t>(offset) * dstStride, dstStride, w, h, buffer_stride, aaf[plane], buffers);
}

SangNom2::SangNom2(PClip _child, int order, int aa, int aac, int threads, bool dh, bool luma, bool chroma, int opt, IScriptEnvironment* env)
    : GenericVideoFilter(_child), _order(order), _dh(dh), aaf{ 0.0f, 0.0f, 0.0f }, processPlane{ luma, chroma, chroma }, bufferLine_(nullptr, aligned_free), bufferPool(nullptr, aligned_free)
{
    has_at_least_v8 = env->FunctionExists("propShow");

    const int _aa[3]{ aa, aac, aac };
    for (int i{ 0 }; i < std::min(vi.NumComponents(), 3); ++i)
        aaf[i] = (vi.ComponentSize() < 4) ? ((_aa[i] * 21.0f / 16.0f) * (1 << (vi.BitsPerComponent() - 8))) : ((_aa[i] * 21.0f / 16.0f) / 256.0f);

    if (_dh)
        vi.height *= 2;

    bufferStride = ((vi.width + alignment - 1) & ~(alignment - 1)) * vi.ComponentSize();
    bufferHeight = (vi.height + 1) >> 1;

    auto aligned_malloc = [&](size_t size, size_t align)
    {
#ifdef _MSC_VER 
        return _aligned_malloc(size, align);
#else 
        void* result;
        if (posix_memalign(&result, align, size))
            return result = nullptr;
        else
            return result;
#endif
    };

    bufferLine_.reset(aligned_malloc(bufferStride * (vi.ComponentSize() != 4 ? 2 : 1), alignment)); // line buffer used in process buffers

    const int bufferPoolSize{ bufferStride * (bufferHeight + 1) * TOTAL_BUFFERS };
    bufferPool.reset(reinterpret_cast<uint8_t*>(aligned_malloc(bufferPoolSize, alignment)));

    // separate bufferpool to multiple pieces
    for (int i{ 0 }; i < TOTAL_BUFFERS; ++i)
        buffers_[i] = bufferPool.get() + i * bufferStride * (bufferHeight + 1);

    if ((!!(env->GetCPUFlags() & CPUF_SSE2) && opt < 0) || opt == 1)
    {
        switch (vi.ComponentSize())
        {
            case 1: process = &SangNom2::sangnom_sse<uint8_t, int16_t>; break;
            case 2: process = &SangNom2::sangnom_sse<uint16_t, int32_t>; break;
            default: process = &SangNom2::sangnom_sse<float, float>; break;
        }
    }
    else
    {
        switch (vi.ComponentSize())
        {
            case 1: process = &SangNom2::sangnom_c<uint8_t, int16_t>; break;
            case 2: process = &SangNom2::sangnom_c<uint16_t, int32_t>; break;
            default: process = &SangNom2::sangnom_c<float, float>; break;
        }
    }
}

PVideoFrame __stdcall SangNom2::GetFrame(int n, IScriptEnvironment* env)
{
    int offset{ 42 }; // Initialise it to shut up a warning.

    switch (_order)
    {
        case as_int(SangNomOrderType::SNOT_DFR): offset = child->GetParity(n) ? 0 : 1; break;
        case as_int(SangNomOrderType::SNOT_SFR_KT): offset = 0; break;
        default: offset = 1; break;
    }

    PVideoFrame src{ child->GetFrame(n, env) };
    PVideoFrame dst{ has_at_least_v8 ? env->NewVideoFrameP(vi, &src, alignment) : env->NewVideoFrame(vi) };

    int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int planecount{ std::min(vi.NumComponents(), 3) };
    for (int i{ 0 }; i < planecount; ++i)
    {
        const int plane{ planes_y[i] };

        auto srcp{ src->GetReadPtr(plane) };
        auto src_stride{ src->GetPitch(plane) };
        auto dst_stride{ dst->GetPitch(plane) };
        auto dstStride{ dst->GetPitch(plane) / vi.ComponentSize() };
        auto src_height{ src->GetHeight(plane) };
        auto dst_height{ dst->GetHeight(plane) };
        auto width{ src->GetRowSize(plane) };
        auto dstp{ dst->GetWritePtr(plane) };

        if (_dh)
        {
            // always process the plane if dh=true
            // copy target field
            env->BitBlt(dstp + static_cast<int64_t>(offset) * dst_stride, dst_stride * 2, srcp, src_stride, width, src_height);
        }
        else
        {
            if (!processPlane[i])
            {
                // copy whole plane
                std::memcpy(dstp, srcp, static_cast<int64_t>(src_stride) * src_height);
                continue;
            }
            // copy target field
            env->BitBlt(dstp + static_cast<int64_t>(offset) * dst_stride, dst_stride * 2, srcp + static_cast<int64_t>(offset) * src_stride, src_stride * 2, width, src_height / 2);
        }

        // copy the field which can't be interpolated
        if (offset == 0)
        {
            // keep top field so the bottom line can't be interpolated
            // just copy the data from its correspond top field
            std::memcpy(dstp + (dst_height - static_cast<int64_t>(1)) * dst_stride, dstp + (dst_height - static_cast<int64_t>(2)) * dst_stride, width);
        }
        else
        {
            // keep bottom field so the top line can't be interpolated
            // just copy the data from its correspond bottom field
            std::memcpy(dstp, dstp + dst_stride, width);
        }

        (this->*process)(dstp, dstStride, width / vi.ComponentSize(), dst_height, offset, i);
    }

    return dst;
}

AVSValue __cdecl Create_SangNom2(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    const VideoInfo& vi = args[0].AsClip()->GetVideoInfo();
    const int _order{ args[1].AsInt(1) };
    const int _aa{ args[2].AsInt(48) };
    const int _aac{ args[3].AsInt(0) };
    const int opt_{ args[8].AsInt(-1) };

    if (vi.IsRGB() || !vi.IsPlanar())
        env->ThrowError("SangNom2: clip must be in Y/YUV planar format.");
    if (vi.height % 2 != 0)
        env->ThrowError("SangNom2: height must be even.");
    if (_order < 0 || _order > 2)
        env->ThrowError("SangNom2: order must be between 0..2.");
    if (_aa < 0 || _aa > 128)
        env->ThrowError("SangNom2: aa must be between 0..128.");
    if (_aac < 0 || _aac > 128)
        env->ThrowError("SangNom2: aac must be between 0..128.");
    if (opt_ < -1 || opt_ > 1)
        env->ThrowError("SangNom2: opt must be between -1..2.");
    if (!(env->GetCPUFlags() & CPUF_SSE2) && opt_ == 1)
        env->ThrowError("SangNom2: opt=1 requires SSE2.");

    return new SangNom2(
        args[0].AsClip(),
        _order,
        _aa,
        _aac,
        args[4].AsInt(0),
        args[5].AsBool(false),
        args[6].AsBool(true),
        args[7].AsBool(true),
        opt_,
        env);
}

AVSValue __cdecl Create_SangNom(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    const VideoInfo& vi = args[0].AsClip()->GetVideoInfo();
    const int _order{ args[1].AsInt(1) };
    static const char ord[3] = { 2, 1, 0 };
    const int _aa{ args[2].AsInt(48) };
    const int _aac{ args[3].AsInt(0) };
    const int opt_{ args[8].AsInt(-1) };

    if (vi.IsRGB() || !vi.IsPlanar())
        env->ThrowError("SangNom: clip must be in Y/YUV planar format.");
    if (vi.height % 2 != 0)
        env->ThrowError("SangNom: height must be even.");
    if (_order < 0 || _order > 2)
        env->ThrowError("SangNom: order must be between 0..2.");
    if (_aa < 0 || _aa > 128)
        env->ThrowError("SangNom: aa must be between 0..128.");
    if (opt_ < -1 || opt_ > 1)
        env->ThrowError("SangNom: opt must be between -1..2.");
    if (!(env->GetCPUFlags() & CPUF_SSE2) && opt_ == 1)
        env->ThrowError("SangNom: opt=1 requires SSE2.");

    return new SangNom2(
        args[0].AsClip(),
        _order != 1 ? ord[_order] : 1,
        _aa,
        _aac,
        args[4].AsInt(0),
        args[5].AsBool(false),
        args[6].AsBool(true),
        args[7].AsBool(true),
        opt_,
        env);
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("SangNom2", "c[order]i[aa]i[aac]i[threads]i[dh]b[luma]b[chroma]b[opt]i", Create_SangNom2, 0);
    env->AddFunction("SangNom", "c[order]i[aa]i[opt]i", Create_SangNom, 0);
    return "SangNom2";
}
