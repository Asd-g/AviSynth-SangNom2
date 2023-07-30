#pragma once

#include <memory>
#include <type_traits>

#include "avisynth.h"

enum class Buffers
{
    ADIFF_M3_P3 = 0,
    ADIFF_M2_P2 = 1,
    ADIFF_M1_P1 = 2,
    ADIFF_P0_M0 = 4,
    ADIFF_P1_M1 = 6,
    ADIFF_P2_M2 = 7,
    ADIFF_P3_M3 = 8,

    SG_FORWARD = 3,
    SG_REVERSE = 5
};

static constexpr int TOTAL_BUFFERS{ 9 };

template <typename Enumeration>
constexpr std::enable_if_t<std::is_enum_v<Enumeration>, std::underlying_type_t<Enumeration>>
as_int(const Enumeration value)
{
    return static_cast<std::underlying_type_t<Enumeration>>(value);
}

static AVS_FORCEINLINE void aligned_free(void* ptr)
{
#ifdef _MSC_VER 
    _aligned_free(ptr);
#else 
    free(ptr);
#endif
}

class SangNom2 : public GenericVideoFilter
{
    int _order;
    bool _dh;
    float aaf[3];
    int bufferStride;
    int bufferHeight;
    bool processPlane[3];
    bool has_at_least_v8;
    std::unique_ptr<void, decltype(&aligned_free)> bufferLine_;
    std::unique_ptr<uint8_t[], decltype(&aligned_free)> bufferPool;
    uint8_t* buffers_[9];

    template <typename T, typename IType>
    void sangnom_sse(uint8_t* __restrict dstp_, const int dstStride, const int w, const int h, int offset, int plane) noexcept;
    template <typename T, typename IType>
    void sangnom_c(uint8_t* __restrict dstp_, const int dstStride, const int w, const int h, int offset, int plane) noexcept;

    void(SangNom2::* process)(uint8_t* __restrict dstp_, const int dstStride, const int w, const int h, int offset, int plane) noexcept;

public:
    SangNom2(PClip _child, int order, int aa, int aac, int threads, bool dh, bool luma, bool chroma, int opt, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    }
};
