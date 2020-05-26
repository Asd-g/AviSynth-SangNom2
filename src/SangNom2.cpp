
/**
 *  SangNom - VapourSynth Single Field Deinterlacer
 *
 *  Copyright (c) 2016 james1201
 *  Copyright (c) 2013 Victor Efimov
 *
 *  This project is licensed under the MIT license. Binaries are GPL v2.
 **/

#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>

#include "avisynth.h"
#include "avs/minmax.h"

#define VS_TARGET_CPU_X86

#ifdef VS_TARGET_CPU_X86
#include <emmintrin.h>
static const size_t sseBytes = 16;
#endif

static const size_t alignment = 32;

enum SangNomOrderType
{
    SNOT_DFR = 0,   // double frame rate, user must call std.SeparateFields().std.DoubleWeave() before use this
    SNOT_SFR_KT,    // single frame rate, keep top field
    SNOT_SFR_KB     // single frame rate, keep bottom field
};

enum Buffers
{
    ADIFF_M3_P3 = 0,
    ADIFF_M2_P2 = 1,
    ADIFF_M1_P1 = 2,
    ADIFF_P0_M0 = 4,
    ADIFF_P1_M1 = 6,
    ADIFF_P2_M2 = 7,
    ADIFF_P3_M3 = 8,

    SG_FORWARD = 3,
    SG_REVERSE = 5,

    TOTAL_BUFFERS = 9,
};

enum class BorderMode
{
    LEFT,
    RIGHT,
    NONE
};

class SangNom2 : public GenericVideoFilter {
    int _order;
    int _aa, _aac;
    bool _dh;
    bool _luma, _chroma;
    float aaf[3];
    int bufferStride;
    int bufferHeight;
    bool processPlane[3];
    bool has_at_least_v8;

    template <typename T, typename IType>
    void sangnom_sse(T* dstp, const int dstStride, const int w, const int h, int offset, int plane, T* buffers[TOTAL_BUFFERS], IType* bufferLine);
    template <typename T, typename IType>
    void sangnom_c(T* dstp, const int dstStride, const int w, const int h, int offset, int plane, T* buffers[TOTAL_BUFFERS], IType* bufferLine);


public:
    SangNom2(PClip _child, int order, int aa, int aac, int threads, bool dh, bool luma, bool chroma, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    }
};

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#ifdef VS_TARGET_CPU_X86

static inline __m128i _mm_packus_epi32_sse2(const __m128i& v1, const __m128i& v2)
{
    __m128i ones = _mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128());
    __m128i subMask32 = _mm_srli_epi32(_mm_slli_epi32(ones, 31), 16);
    __m128i addMask16 = _mm_slli_epi16(ones, 15);
    return _mm_add_epi16(_mm_packs_epi32(_mm_sub_epi32(v1, subMask32), _mm_sub_epi32(v2, subMask32)), addMask16);
}

static inline __m128i convSign16(const __m128i& v1)
{
    alignas(sizeof(__m128i)) static const uint16_t signMask16[8] = { 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000 };
    return _mm_xor_si128(v1, _mm_load_si128(reinterpret_cast<const __m128i*>(signMask16)));
}

template <typename T, bool aligned>
static inline __m128i sse_load_si128(const T* ptr)
{
    if (aligned)
        return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
}

template <typename T, bool aligned>
static inline __m128 sse_load_ps(const T* ptr)
{
    if (aligned)
        return _mm_load_ps(ptr);
    return _mm_loadu_ps(ptr);
}

template <typename T, bool aligned>
static inline void sse_store_si128(T* ptr, __m128i val)
{
    if (aligned)
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr), val);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), val);
}

template <typename T, bool aligned>
static inline void sse_store_ps(T* ptr, __m128 val)
{
    if (aligned)
        _mm_store_ps(ptr, val);
    _mm_storeu_ps(ptr, val);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i sse_load_1_to_left_si128(const T* ptr)
{
    static_assert(std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4),
        "error instantiation, T must be 8, 16, 32 bit int");

    if (isBorder) {

        if (sizeof(T) == 1) {
            auto mask = _mm_setr_epi8(0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_slli_si128(val, 1);
            auto andm = _mm_and_si128(val, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 2) {
            auto mask = _mm_setr_epi16(0xFFFF, 00, 00, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_slli_si128(val, 2);
            auto andm = _mm_and_si128(val, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 4) {
            auto mask = _mm_setr_epi32(0xFFFFFFFF, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_slli_si128(val, 4);
            auto andm = _mm_and_si128(val, mask);
            return _mm_or_si128(shifted, andm);
        }
    }

    return sse_load_si128<T, false>(ptr - 1);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128 sse_load_1_to_left_ps(const T* ptr)
{
    static_assert(std::is_floating_point<T>::value && (sizeof(T) == 4),
        "error instantiation, T must be 32 bit float");

    if (isBorder) {

        if (sizeof(T) == 4) {
            auto val = sse_load_ps<T, alignedLoad>(ptr);
            return _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 1, 0, 0));
        }
    }

    return sse_load_ps<T, false>(ptr - 1);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i sse_load_2_to_left_si128(const T* ptr)
{
    static_assert(std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4),
        "error instantiation, T must be 8, 16, 32 bit int");

    if (isBorder) {

        if (sizeof(T) == 1) {
            auto mask = _mm_setr_epi8(0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_slli_si128(val, 2);
            auto unpck = _mm_unpacklo_epi8(val, val);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 2) {
            auto mask = _mm_setr_epi16(0xFFFF, 0xFFFF, 00, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_slli_si128(val, 4);
            auto unpck = _mm_unpacklo_epi16(val, val);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 4) {
            auto mask = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_slli_si128(val, 8);
            auto unpck = _mm_unpacklo_epi32(val, val);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
    }

    return sse_load_si128<T, false>(ptr - 2);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128 sse_load_2_to_left_ps(const T* ptr)
{
    static_assert(std::is_floating_point<T>::value && (sizeof(T) == 4),
        "error instantiation, T must be 32 bit float");

    if (isBorder) {

        if (sizeof(T) == 4) {
            auto val = sse_load_ps<T, alignedLoad>(ptr);
            return _mm_shuffle_ps(val, val, _MM_SHUFFLE(1, 0, 0, 0));
        }
    }

    return sse_load_ps<T, false>(ptr - 2);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i sse_load_3_to_left_si128(const T* ptr)
{
    static_assert(std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4),
        "error instantiation, T must be 8, 16, 32 bit int");

    if (isBorder) {

        if (sizeof(T) == 1) {
            auto mask = _mm_setr_epi8(0xFF, 0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_slli_si128(val, 3);
            auto unpck = _mm_unpacklo_epi8(val, val);
            unpck = _mm_unpacklo_epi16(unpck, unpck);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 2) {
            auto mask = _mm_setr_epi16(0xFFFF, 0xFFFF, 0xFFFF, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_slli_si128(val, 6);
            auto unpck = _mm_unpacklo_epi16(val, val);
            unpck = _mm_unpacklo_epi32(unpck, unpck);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 4) {
            auto mask = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_slli_si128(val, 12);
            auto unpck = _mm_unpacklo_epi32(val, val);
            unpck = _mm_unpacklo_epi64(unpck, unpck);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
    }

    return sse_load_si128<T, false>(ptr - 3);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128 sse_load_3_to_left_ps(const T* ptr)
{
    static_assert(std::is_floating_point<T>::value && (sizeof(T) == 4),
        "error instantiation, T must be 32 bit float");

    if (isBorder) {

        if (sizeof(T) == 4) {
            return _mm_set1_ps(ptr[0]);
        }
    }

    return sse_load_ps<T, false>(ptr - 3);
}

//note the difference between set and setr for left and right loading
template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i sse_load_1_to_right_si128(const T* ptr)
{
    static_assert(std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4),
        "error instantiation, T must be 8, 16, 32 bit int");

    if (isBorder) {

        if (sizeof(T) == 1) {
            auto mask = _mm_set_epi8(0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_srli_si128(val, 1);
            auto andm = _mm_and_si128(val, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 2) {
            auto mask = _mm_set_epi16(0xFFFF, 00, 00, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_srli_si128(val, 2);
            auto andm = _mm_and_si128(val, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 4) {
            auto mask = _mm_set_epi32(0xFFFFFFFF, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_srli_si128(val, 4);
            auto andm = _mm_and_si128(val, mask);
            return _mm_or_si128(shifted, andm);
        }
    }

    return sse_load_si128<T, false>(ptr + 1);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128 sse_load_1_to_right_ps(const T* ptr)
{
    static_assert(std::is_floating_point<T>::value && (sizeof(T) == 4),
        "error instantiation, T must be 32 bit float");

    if (isBorder) {

        if (sizeof(T) == 4) {
            auto val = sse_load_ps<T, alignedLoad>(ptr);
            return _mm_shuffle_ps(val, val, _MM_SHUFFLE(3, 3, 2, 1));
        }
    }

    return sse_load_ps<T, false>(ptr + 1);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i sse_load_2_to_right_si128(const T* ptr)
{
    static_assert(std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4),
        "error instantiation, T must be 8, 16, 32 bit int");

    if (isBorder) {

        if (sizeof(T) == 1) {
            auto mask = _mm_set_epi8(0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_srli_si128(val, 2);
            auto unpck = _mm_unpackhi_epi8(val, val);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 2) {
            auto mask = _mm_set_epi16(0xFFFF, 0xFFFF, 00, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_srli_si128(val, 4);
            auto unpck = _mm_unpackhi_epi16(val, val);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 4) {
            auto mask = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_srli_si128(val, 8);
            auto unpck = _mm_unpackhi_epi32(val, val);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
    }

    return sse_load_si128<T, false>(ptr + 2);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128 sse_load_2_to_right_ps(const T* ptr)
{
    static_assert(std::is_floating_point<T>::value && (sizeof(T) == 4),
        "error instantiation, T must be 32 bit float");

    if (isBorder) {

        if (sizeof(T) == 4) {
            auto val = sse_load_ps<T, alignedLoad>(ptr);
            return _mm_shuffle_ps(val, val, _MM_SHUFFLE(3, 3, 3, 2));
        }
    }

    return sse_load_ps<T, false>(ptr + 2);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128i sse_load_3_to_right_si128(const T* ptr)
{
    static_assert(std::is_integral<T>::value && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4),
        "error instantiation, T must be 8, 16, 32 bit int");

    if (isBorder) {

        if (sizeof(T) == 1) {
            auto mask = _mm_set_epi8(0xFF, 0xFF, 0xFF, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_srli_si128(val, 3);
            auto unpck = _mm_unpackhi_epi8(val, val);
            unpck = _mm_unpackhi_epi16(unpck, unpck);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 2) {
            auto mask = _mm_set_epi16(0xFFFF, 0xFFFF, 0xFFFF, 00, 00, 00, 00, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_srli_si128(val, 6);
            auto unpck = _mm_unpackhi_epi16(val, val);
            unpck = _mm_unpackhi_epi32(unpck, unpck);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
        else if (sizeof(T) == 4) {
            auto mask = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 00);
            auto val = sse_load_si128<T, alignedLoad>(ptr);
            auto shifted = _mm_srli_si128(val, 12);
            auto unpck = _mm_unpackhi_epi32(val, val);
            unpck = _mm_unpackhi_epi64(unpck, unpck);
            auto andm = _mm_and_si128(unpck, mask);
            return _mm_or_si128(shifted, andm);
        }
    }

    return sse_load_si128<T, false>(ptr + 3);
}

template <typename T, bool isBorder, bool alignedLoad>
static inline __m128 sse_load_3_to_right_ps(const T* ptr)
{
    static_assert(std::is_floating_point<T>::value && (sizeof(T) == 4),
        "error instantiation, T must be 32 bit float");

    if (isBorder) {

        if (sizeof(T) == 4) {
            return _mm_set1_ps(ptr[3]);
        }
    }

    return sse_load_ps<T, false>(ptr + 3);
}

template <typename T, typename Arg>
static inline Arg sse_abs_diff(Arg a, Arg b);

template <>
inline __m128i sse_abs_diff<uint8_t, __m128i>(__m128i a, __m128i b)
{
    auto positive = _mm_subs_epu8(a, b);
    auto negative = _mm_subs_epu8(b, a);
    return _mm_or_si128(positive, negative);
}

template <>
inline __m128i sse_abs_diff<uint16_t, __m128i>(__m128i a, __m128i b)
{
    auto positive = _mm_subs_epu16(a, b);
    auto negative = _mm_subs_epu16(b, a);
    return _mm_or_si128(positive, negative);
}

template <>
inline __m128 sse_abs_diff<float, __m128>(__m128 a, __m128 b)
{
    auto positive = _mm_sub_ps(a, b);
    auto negative = _mm_sub_ps(b, a);
    return _mm_max_ps(positive, negative);
}

template <typename T, typename Arg>
inline Arg calculateSangNom_sse(const Arg& p1, const Arg& p2, const Arg& p3);

template <>
inline __m128i calculateSangNom_sse<uint8_t, __m128i>(const __m128i& p1, const __m128i& p2, const __m128i& p3)
{
    const auto const_0 = _mm_setzero_si128();

    auto p1_lo = _mm_unpacklo_epi8(p1, const_0);
    auto p1_hi = _mm_unpackhi_epi8(p1, const_0);

    auto p2_lo = _mm_unpacklo_epi8(p2, const_0);
    auto p2_hi = _mm_unpackhi_epi8(p2, const_0);

    auto p3_lo = _mm_unpacklo_epi8(p3, const_0);
    auto p3_hi = _mm_unpackhi_epi8(p3, const_0);

    p1_lo = _mm_slli_epi16(p1_lo, 2); // p1 * 4
    p1_hi = _mm_slli_epi16(p1_hi, 2);

    auto sum_lo = _mm_add_epi16(p1_lo, p2_lo); // p1 * 4 + p2
    auto sum_hi = _mm_add_epi16(p1_hi, p2_hi);

    p2_lo = _mm_slli_epi16(p2_lo, 2);
    p2_hi = _mm_slli_epi16(p2_hi, 2);

    sum_lo = _mm_add_epi16(sum_lo, p2_lo); // p1 * 4 + p2 * 5
    sum_hi = _mm_add_epi16(sum_hi, p2_hi);

    sum_lo = _mm_sub_epi16(sum_lo, p3_lo); // p1 * 4 + p2 * 5 - p3
    sum_hi = _mm_sub_epi16(sum_hi, p3_hi);

    sum_lo = _mm_srli_epi16(sum_lo, 3); // (p1 * 4 + p2 * 5 - p3) / 8
    sum_hi = _mm_srli_epi16(sum_hi, 3);

    return _mm_packus_epi16(sum_lo, sum_hi);
}

template <>
inline __m128i calculateSangNom_sse<uint16_t, __m128i>(const __m128i& p1, const __m128i& p2, const __m128i& p3)
{
    const auto const_0 = _mm_setzero_si128();

    auto p1_lo = _mm_unpacklo_epi16(p1, const_0);
    auto p1_hi = _mm_unpackhi_epi16(p1, const_0);

    auto p2_lo = _mm_unpacklo_epi16(p2, const_0);
    auto p2_hi = _mm_unpackhi_epi16(p2, const_0);

    auto p3_lo = _mm_unpacklo_epi16(p3, const_0);
    auto p3_hi = _mm_unpackhi_epi16(p3, const_0);

    p1_lo = _mm_slli_epi32(p1_lo, 2); // p1 * 4
    p1_hi = _mm_slli_epi32(p1_hi, 2);

    auto sum_lo = _mm_add_epi32(p1_lo, p2_lo); // p1 * 4 + p2
    auto sum_hi = _mm_add_epi32(p1_hi, p2_hi);

    p2_lo = _mm_slli_epi32(p2_lo, 2);
    p2_hi = _mm_slli_epi32(p2_hi, 2);

    sum_lo = _mm_add_epi32(sum_lo, p2_lo); // p1 * 4 + p2 * 5
    sum_hi = _mm_add_epi32(sum_hi, p2_hi);

    sum_lo = _mm_sub_epi32(sum_lo, p3_lo); // p1 * 4 + p2 * 5 - p3
    sum_hi = _mm_sub_epi32(sum_hi, p3_hi);

    sum_lo = _mm_srli_epi32(sum_lo, 3); // (p1 * 4 + p2 * 5 - p3) / 8
    sum_hi = _mm_srli_epi32(sum_hi, 3);

    return _mm_packus_epi32_sse2(sum_lo, sum_hi);
}

template <>
inline __m128 calculateSangNom_sse<float, __m128>(const __m128& p1, const __m128& p2, const __m128& p3)
{
    const auto const_4 = _mm_set1_ps(4.0);
    const auto const_5 = _mm_set1_ps(5.0);
    const auto const_1_8 = _mm_set1_ps(1.0 / 8.0);

    auto p1x4 = _mm_mul_ps(p1, const_4);
    auto p2x5 = _mm_mul_ps(p2, const_5);

    auto sum = _mm_add_ps(p1x4, p2x5);
    sum = _mm_sub_ps(sum, p3);

    sum = _mm_mul_ps(sum, const_1_8); // (p1 * 4 + p2 * 5 - p3) / 8

    return sum;
}

template <typename T, typename Arg>
static inline Arg getAvgIfMinBuf(const Arg& a1, const Arg& a2, const Arg& buf, const Arg& minBuf, const Arg& acc);

template <>
inline __m128i getAvgIfMinBuf<uint8_t, __m128i>(const __m128i& a1, const __m128i& a2, const __m128i& buf, const __m128i& minBuf, const __m128i& acc)
{
    auto avg = _mm_avg_epu8(a1, a2);
    auto mask = _mm_cmpeq_epi8(buf, minBuf);
    auto avgPart = _mm_and_si128(mask, avg);
    auto accPart = _mm_andnot_si128(mask, acc);
    return _mm_or_si128(avgPart, accPart);
}

template <>
inline __m128i getAvgIfMinBuf<uint16_t, __m128i>(const __m128i& a1, const __m128i& a2, const __m128i& buf, const __m128i& minBuf, const __m128i& acc)
{
    auto avg = _mm_avg_epu16(a1, a2);
    auto mask = _mm_cmpeq_epi16(buf, minBuf);
    auto avgPart = _mm_and_si128(mask, avg);
    auto accPart = _mm_andnot_si128(mask, acc);
    return _mm_or_si128(avgPart, accPart);
}

template <>
inline __m128 getAvgIfMinBuf<float, __m128>(const __m128& a1, const __m128& a2, const __m128& buf, const __m128& minBuf, const __m128& acc)
{
    const auto const_1_2 = _mm_set1_ps(1.0 / 2.0);
    auto avg = _mm_mul_ps(_mm_add_ps(a1, a2), const_1_2);
    auto mask = _mm_cmple_ps(buf, minBuf);
    auto avgPart = _mm_and_ps(mask, avg);
    auto accPart = _mm_andnot_ps(mask, acc);
    return _mm_or_ps(avgPart, accPart);
}

#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename IType>
static inline IType loadPixel(const T* srcp, int curPos, int offset, int width)
{
    int reqPos = curPos + offset;
    if (reqPos >= 0 && reqPos < width)
        return srcp[reqPos];
    if (reqPos >= 0)
        return srcp[width - 1];
    return srcp[0];
}

template <typename T>
static inline T absDiff(const T& a, const T& b)
{
    return std::abs(a - b);
}

template <>
inline float absDiff<float>(const float& a, const float& b)
{
    return std::fabs(a - b);
}

template <typename T, typename IType>
static inline IType calculateSangNom(const T& p1, const T& p2, const T& p3)
{
    IType sum = p1 * 5 + p2 * 4 - p3;
    return sum / 8;
}

#ifdef VS_TARGET_CPU_X86

template <BorderMode border, bool alignedLoad, bool alignedStore>
static inline void prepareBuffersLine_sse(const uint8_t* srcp, const uint8_t* srcpn2, uint8_t* buffers[TOTAL_BUFFERS], const int w, const int bufferOffset)
{
    constexpr int pixelStep = sseBytes / sizeof(uint8_t);

    for (int x = 0; x < w; x += pixelStep) {

        auto currLineM3 = sse_load_3_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM2 = sse_load_2_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM1 = sse_load_1_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLine = sse_load_si128<uint8_t, alignedLoad>(srcp + x);
        auto currLineP1 = sse_load_1_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP2 = sse_load_2_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP3 = sse_load_3_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);

        auto nextLineM3 = sse_load_3_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM2 = sse_load_2_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM1 = sse_load_1_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLine = sse_load_si128<uint8_t, alignedLoad>(srcpn2 + x);
        auto nextLineP1 = sse_load_1_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP2 = sse_load_2_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP3 = sse_load_3_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);

        auto forwardSangNom1 = calculateSangNom_sse<uint8_t, __m128i>(currLineM1, currLine, currLineP1);
        auto forwardSangNom2 = calculateSangNom_sse<uint8_t, __m128i>(nextLineP1, nextLine, nextLineM1);
        auto backwardSangNom1 = calculateSangNom_sse<uint8_t, __m128i>(currLineP1, currLine, currLineM1);
        auto backwardSangNom2 = calculateSangNom_sse<uint8_t, __m128i>(nextLineM1, nextLine, nextLineP1);

        sse_store_si128<uint8_t, alignedStore>((buffers[ADIFF_M3_P3] + bufferOffset + x), sse_abs_diff<uint8_t, __m128i>(currLineM3, nextLineP3));
        sse_store_si128<uint8_t, alignedStore>((buffers[ADIFF_M2_P2] + bufferOffset + x), sse_abs_diff<uint8_t, __m128i>(currLineM2, nextLineP2));
        sse_store_si128<uint8_t, alignedStore>((buffers[ADIFF_M1_P1] + bufferOffset + x), sse_abs_diff<uint8_t, __m128i>(currLineM1, nextLineP1));
        sse_store_si128<uint8_t, alignedStore>((buffers[ADIFF_P0_M0] + bufferOffset + x), sse_abs_diff<uint8_t, __m128i>(currLine, nextLine));
        sse_store_si128<uint8_t, alignedStore>((buffers[ADIFF_P1_M1] + bufferOffset + x), sse_abs_diff<uint8_t, __m128i>(currLineP1, nextLineM1));
        sse_store_si128<uint8_t, alignedStore>((buffers[ADIFF_P2_M2] + bufferOffset + x), sse_abs_diff<uint8_t, __m128i>(currLineP2, nextLineM2));
        sse_store_si128<uint8_t, alignedStore>((buffers[ADIFF_P3_M3] + bufferOffset + x), sse_abs_diff<uint8_t, __m128i>(currLineP3, nextLineM3));

        sse_store_si128<uint8_t, alignedStore>((buffers[SG_FORWARD] + bufferOffset + x), sse_abs_diff<uint8_t, __m128i>(forwardSangNom1, forwardSangNom2));
        sse_store_si128<uint8_t, alignedStore>((buffers[SG_REVERSE] + bufferOffset + x), sse_abs_diff<uint8_t, __m128i>(backwardSangNom1, backwardSangNom2));
    }
}

template <BorderMode border, bool alignedLoad, bool alignedStore>
static inline void prepareBuffersLine_sse(const uint16_t* srcp, const uint16_t* srcpn2, uint16_t* buffers[TOTAL_BUFFERS], const int w, const int bufferOffset)
{
    constexpr int pixelStep = sseBytes / sizeof(uint16_t);

    for (int x = 0; x < w; x += pixelStep) {

        auto currLineM3 = sse_load_3_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM2 = sse_load_2_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM1 = sse_load_1_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLine = sse_load_si128<uint16_t, alignedLoad>(srcp + x);
        auto currLineP1 = sse_load_1_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP2 = sse_load_2_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP3 = sse_load_3_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);

        auto nextLineM3 = sse_load_3_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM2 = sse_load_2_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM1 = sse_load_1_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLine = sse_load_si128<uint16_t, alignedLoad>(srcpn2 + x);
        auto nextLineP1 = sse_load_1_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP2 = sse_load_2_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP3 = sse_load_3_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);

        auto forwardSangNom1 = calculateSangNom_sse<uint16_t, __m128i>(currLineM1, currLine, currLineP1);
        auto forwardSangNom2 = calculateSangNom_sse<uint16_t, __m128i>(nextLineP1, nextLine, nextLineM1);
        auto backwardSangNom1 = calculateSangNom_sse<uint16_t, __m128i>(currLineP1, currLine, currLineM1);
        auto backwardSangNom2 = calculateSangNom_sse<uint16_t, __m128i>(nextLineM1, nextLine, nextLineP1);

        sse_store_si128<uint16_t, alignedStore>((buffers[ADIFF_M3_P3] + bufferOffset + x), sse_abs_diff<uint16_t, __m128i>(currLineM3, nextLineP3));
        sse_store_si128<uint16_t, alignedStore>((buffers[ADIFF_M2_P2] + bufferOffset + x), sse_abs_diff<uint16_t, __m128i>(currLineM2, nextLineP2));
        sse_store_si128<uint16_t, alignedStore>((buffers[ADIFF_M1_P1] + bufferOffset + x), sse_abs_diff<uint16_t, __m128i>(currLineM1, nextLineP1));
        sse_store_si128<uint16_t, alignedStore>((buffers[ADIFF_P0_M0] + bufferOffset + x), sse_abs_diff<uint16_t, __m128i>(currLine, nextLine));
        sse_store_si128<uint16_t, alignedStore>((buffers[ADIFF_P1_M1] + bufferOffset + x), sse_abs_diff<uint16_t, __m128i>(currLineP1, nextLineM1));
        sse_store_si128<uint16_t, alignedStore>((buffers[ADIFF_P2_M2] + bufferOffset + x), sse_abs_diff<uint16_t, __m128i>(currLineP2, nextLineM2));
        sse_store_si128<uint16_t, alignedStore>((buffers[ADIFF_P3_M3] + bufferOffset + x), sse_abs_diff<uint16_t, __m128i>(currLineP3, nextLineM3));

        sse_store_si128<uint16_t, alignedStore>((buffers[SG_FORWARD] + bufferOffset + x), sse_abs_diff<uint16_t, __m128i>(forwardSangNom1, forwardSangNom2));
        sse_store_si128<uint16_t, alignedStore>((buffers[SG_REVERSE] + bufferOffset + x), sse_abs_diff<uint16_t, __m128i>(backwardSangNom1, backwardSangNom2));
    }
}

template <BorderMode border, bool alignedLoad, bool alignedStore>
static inline void prepareBuffersLine_sse(const float* srcp, const float* srcpn2, float* buffers[TOTAL_BUFFERS], const int w, const int bufferOffset)
{
    constexpr int pixelStep = sseBytes / sizeof(float);

    for (int x = 0; x < w; x += pixelStep) {

        auto currLineM3 = sse_load_3_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM2 = sse_load_2_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM1 = sse_load_1_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLine = sse_load_ps<float, alignedLoad>(srcp + x);
        auto currLineP1 = sse_load_1_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP2 = sse_load_2_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP3 = sse_load_3_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcp + x);

        auto nextLineM3 = sse_load_3_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM2 = sse_load_2_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM1 = sse_load_1_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLine = sse_load_ps<float, alignedLoad>(srcpn2 + x);
        auto nextLineP1 = sse_load_1_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP2 = sse_load_2_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP3 = sse_load_3_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);

        auto forwardSangNom1 = calculateSangNom_sse<float, __m128>(currLineM1, currLine, currLineP1);
        auto forwardSangNom2 = calculateSangNom_sse<float, __m128>(nextLineP1, nextLine, nextLineM1);
        auto backwardSangNom1 = calculateSangNom_sse<float, __m128>(currLineP1, currLine, currLineM1);
        auto backwardSangNom2 = calculateSangNom_sse<float, __m128>(nextLineM1, nextLine, nextLineP1);

        sse_store_ps<float, alignedStore>((buffers[ADIFF_M3_P3] + bufferOffset + x), sse_abs_diff<float, __m128>(currLineM3, nextLineP3));
        sse_store_ps<float, alignedStore>((buffers[ADIFF_M2_P2] + bufferOffset + x), sse_abs_diff<float, __m128>(currLineM2, nextLineP2));
        sse_store_ps<float, alignedStore>((buffers[ADIFF_M1_P1] + bufferOffset + x), sse_abs_diff<float, __m128>(currLineM1, nextLineP1));
        sse_store_ps<float, alignedStore>((buffers[ADIFF_P0_M0] + bufferOffset + x), sse_abs_diff<float, __m128>(currLine, nextLine));
        sse_store_ps<float, alignedStore>((buffers[ADIFF_P1_M1] + bufferOffset + x), sse_abs_diff<float, __m128>(currLineP1, nextLineM1));
        sse_store_ps<float, alignedStore>((buffers[ADIFF_P2_M2] + bufferOffset + x), sse_abs_diff<float, __m128>(currLineP2, nextLineM2));
        sse_store_ps<float, alignedStore>((buffers[ADIFF_P3_M3] + bufferOffset + x), sse_abs_diff<float, __m128>(currLineP3, nextLineM3));

        sse_store_ps<float, alignedStore>((buffers[SG_FORWARD] + bufferOffset + x), sse_abs_diff<float, __m128>(forwardSangNom1, forwardSangNom2));
        sse_store_ps<float, alignedStore>((buffers[SG_REVERSE] + bufferOffset + x), sse_abs_diff<float, __m128>(backwardSangNom1, backwardSangNom2));
    }
}

template <typename T, typename IType>
static inline void prepareBuffers_sse(const T* srcp, const int srcStride, const int w, const int h, const int bufferStride, T* buffers[TOTAL_BUFFERS])
{
    auto srcpn2 = srcp + srcStride * static_cast<int64_t>(2);

    int bufferOffset = bufferStride;

    constexpr int pixelStep = sseBytes / sizeof(T);
    const int wMod = (w - 1) & ~(pixelStep - 1);

    for (int y = 0; y < h / 2 - 1; ++y) {

        prepareBuffersLine_sse<BorderMode::LEFT, true, true>(srcp, srcpn2, buffers, pixelStep, bufferOffset);
        prepareBuffersLine_sse<BorderMode::NONE, true, true>(srcp + pixelStep, srcpn2 + pixelStep, buffers, wMod - pixelStep, bufferOffset + pixelStep);
        prepareBuffersLine_sse<BorderMode::RIGHT, false, false>(srcp + w - pixelStep, srcpn2 + w - pixelStep, buffers, pixelStep, bufferOffset + w - pixelStep);

        srcp += srcStride * static_cast<int64_t>(2);
        srcpn2 += srcStride * static_cast<int64_t>(2);
        bufferOffset += bufferStride;
    }
}

#endif

template <typename T, typename IType>
static inline void prepareBuffers_c(const T* srcp, const int srcStride, const int w, const int h, const int bufferStride, T* buffers[TOTAL_BUFFERS])
{
    auto srcpn2 = srcp + srcStride * static_cast<int64_t>(2);

    int bufferOffset = bufferStride;

    for (int y = 0; y < h / 2 - 1; ++y) {

        for (int x = 0; x < w; ++x) {

            const IType currLineM3 = loadPixel<T, IType>(srcp, x, -3, w);
            const IType currLineM2 = loadPixel<T, IType>(srcp, x, -2, w);
            const IType currLineM1 = loadPixel<T, IType>(srcp, x, -1, w);
            const IType currLine = srcp[x];
            const IType currLineP1 = loadPixel<T, IType>(srcp, x, 1, w);
            const IType currLineP2 = loadPixel<T, IType>(srcp, x, 2, w);
            const IType currLineP3 = loadPixel<T, IType>(srcp, x, 3, w);

            const IType nextLineM3 = loadPixel<T, IType>(srcpn2, x, -3, w);
            const IType nextLineM2 = loadPixel<T, IType>(srcpn2, x, -2, w);
            const IType nextLineM1 = loadPixel<T, IType>(srcpn2, x, -1, w);
            const IType nextLine = srcpn2[x];
            const IType nextLineP1 = loadPixel<T, IType>(srcpn2, x, 1, w);
            const IType nextLineP2 = loadPixel<T, IType>(srcpn2, x, 2, w);
            const IType nextLineP3 = loadPixel<T, IType>(srcpn2, x, 3, w);

            const IType forwardSangNom1 = calculateSangNom<T, IType>(currLineM1, currLine, currLineP1);
            const IType forwardSangNom2 = calculateSangNom<T, IType>(nextLineP1, nextLine, nextLineM1);
            const IType backwardSangNom1 = calculateSangNom<T, IType>(currLineP1, currLine, currLineM1);
            const IType backwardSangNom2 = calculateSangNom<T, IType>(nextLineM1, nextLine, nextLineP1);

            buffers[ADIFF_M3_P3][bufferOffset + x] = static_cast<T>(absDiff(currLineM3, nextLineP3));
            buffers[ADIFF_M2_P2][bufferOffset + x] = static_cast<T>(absDiff(currLineM2, nextLineP2));
            buffers[ADIFF_M1_P1][bufferOffset + x] = static_cast<T>(absDiff(currLineM1, nextLineP1));
            buffers[ADIFF_P0_M0][bufferOffset + x] = static_cast<T>(absDiff(currLine, nextLine));
            buffers[ADIFF_P1_M1][bufferOffset + x] = static_cast<T>(absDiff(currLineP1, nextLineM1));
            buffers[ADIFF_P2_M2][bufferOffset + x] = static_cast<T>(absDiff(currLineP2, nextLineM2));
            buffers[ADIFF_P3_M3][bufferOffset + x] = static_cast<T>(absDiff(currLineP3, nextLineM3));

            buffers[SG_FORWARD][bufferOffset + x] = static_cast<T>(absDiff(forwardSangNom1, forwardSangNom2));
            buffers[SG_REVERSE][bufferOffset + x] = static_cast<T>(absDiff(backwardSangNom1, backwardSangNom2));
        }

        srcp += srcStride * static_cast<int64_t>(2);
        srcpn2 += static_cast<int64_t>(srcStride) * 2;
        bufferOffset += bufferStride;
    }
}

#ifdef VS_TARGET_CPU_X86

template <BorderMode border>
static inline void processBuffersBlock_sse(uint8_t* bufferp, const int16_t* bufferLine, const int x)
{
    constexpr int pixelStep2 = sseBytes / sizeof(int16_t);

    auto currLineM3_lo = sse_load_3_to_left_si128<int16_t, border == BorderMode::LEFT, true>(bufferLine + x);
    auto currLineM2_lo = sse_load_2_to_left_si128<int16_t, border == BorderMode::LEFT, true>(bufferLine + x);
    auto currLineM1_lo = sse_load_1_to_left_si128<int16_t, border == BorderMode::LEFT, true>(bufferLine + x);
    auto currLine_lo = sse_load_si128<int16_t, true>(bufferLine + x);
    auto currLineP1_lo = sse_load_1_to_right_si128<int16_t, false, true>(bufferLine + x);
    auto currLineP2_lo = sse_load_2_to_right_si128<int16_t, false, true>(bufferLine + x);
    auto currLineP3_lo = sse_load_3_to_right_si128<int16_t, false, true>(bufferLine + x);

    auto currLineM3_hi = sse_load_3_to_left_si128<int16_t, false, true>(bufferLine + x + pixelStep2);
    auto currLineM2_hi = sse_load_2_to_left_si128<int16_t, false, true>(bufferLine + x + pixelStep2);
    auto currLineM1_hi = sse_load_1_to_left_si128<int16_t, false, true>(bufferLine + x + pixelStep2);
    auto currLine_hi = sse_load_si128<int16_t, true>(bufferLine + x + pixelStep2);
    auto currLineP1_hi = sse_load_1_to_right_si128<int16_t, border == BorderMode::RIGHT, true>(bufferLine + x + pixelStep2);
    auto currLineP2_hi = sse_load_2_to_right_si128<int16_t, border == BorderMode::RIGHT, true>(bufferLine + x + pixelStep2);
    auto currLineP3_hi = sse_load_3_to_right_si128<int16_t, border == BorderMode::RIGHT, true>(bufferLine + x + pixelStep2);

    auto sum_lo = _mm_add_epi16(currLineM3_lo, currLineM2_lo);
    sum_lo = _mm_add_epi16(sum_lo, currLineM1_lo);
    sum_lo = _mm_add_epi16(sum_lo, currLine_lo);
    sum_lo = _mm_add_epi16(sum_lo, currLineP1_lo);
    sum_lo = _mm_add_epi16(sum_lo, currLineP2_lo);
    sum_lo = _mm_add_epi16(sum_lo, currLineP3_lo);

    sum_lo = _mm_srli_epi16(sum_lo, 4);


    auto sum_hi = _mm_add_epi16(currLineM3_hi, currLineM2_hi);
    sum_hi = _mm_add_epi16(sum_hi, currLineM1_hi);
    sum_hi = _mm_add_epi16(sum_hi, currLine_hi);
    sum_hi = _mm_add_epi16(sum_hi, currLineP1_hi);
    sum_hi = _mm_add_epi16(sum_hi, currLineP2_hi);
    sum_hi = _mm_add_epi16(sum_hi, currLineP3_hi);

    sum_hi = _mm_srli_epi16(sum_hi, 4);


    auto result = _mm_packus_epi16(sum_lo, sum_hi);

    sse_store_si128<uint8_t, true>(bufferp + x, result);
}

template <BorderMode border>
static inline void processBuffersBlock_sse(uint16_t* bufferp, const int32_t* bufferLine, const int x)
{
    constexpr int pixelStep2 = sseBytes / sizeof(int32_t);

    auto currLineM3_lo = sse_load_3_to_left_si128<int32_t, border == BorderMode::LEFT, true>(bufferLine + x);
    auto currLineM2_lo = sse_load_2_to_left_si128<int32_t, border == BorderMode::LEFT, true>(bufferLine + x);
    auto currLineM1_lo = sse_load_1_to_left_si128<int32_t, border == BorderMode::LEFT, true>(bufferLine + x);
    auto currLine_lo = sse_load_si128<int32_t, true>(bufferLine + x);
    auto currLineP1_lo = sse_load_1_to_right_si128<int32_t, false, true>(bufferLine + x);
    auto currLineP2_lo = sse_load_2_to_right_si128<int32_t, false, true>(bufferLine + x);
    auto currLineP3_lo = sse_load_3_to_right_si128<int32_t, false, true>(bufferLine + x);

    auto currLineM3_hi = sse_load_3_to_left_si128<int32_t, false, true>(bufferLine + x + pixelStep2);
    auto currLineM2_hi = sse_load_2_to_left_si128<int32_t, false, true>(bufferLine + x + pixelStep2);
    auto currLineM1_hi = sse_load_1_to_left_si128<int32_t, false, true>(bufferLine + x + pixelStep2);
    auto currLine_hi = sse_load_si128<int32_t, true>(bufferLine + x + pixelStep2);
    auto currLineP1_hi = sse_load_1_to_right_si128<int32_t, border == BorderMode::RIGHT, true>(bufferLine + x + pixelStep2);
    auto currLineP2_hi = sse_load_2_to_right_si128<int32_t, border == BorderMode::RIGHT, true>(bufferLine + x + pixelStep2);
    auto currLineP3_hi = sse_load_3_to_right_si128<int32_t, border == BorderMode::RIGHT, true>(bufferLine + x + pixelStep2);

    auto sum_lo = _mm_add_epi32(currLineM3_lo, currLineM2_lo);
    sum_lo = _mm_add_epi32(sum_lo, currLineM1_lo);
    sum_lo = _mm_add_epi32(sum_lo, currLine_lo);
    sum_lo = _mm_add_epi32(sum_lo, currLineP1_lo);
    sum_lo = _mm_add_epi32(sum_lo, currLineP2_lo);
    sum_lo = _mm_add_epi32(sum_lo, currLineP3_lo);

    sum_lo = _mm_srli_epi32(sum_lo, 4);


    auto sum_hi = _mm_add_epi32(currLineM3_hi, currLineM2_hi);
    sum_hi = _mm_add_epi32(sum_hi, currLineM1_hi);
    sum_hi = _mm_add_epi32(sum_hi, currLine_hi);
    sum_hi = _mm_add_epi32(sum_hi, currLineP1_hi);
    sum_hi = _mm_add_epi32(sum_hi, currLineP2_hi);
    sum_hi = _mm_add_epi32(sum_hi, currLineP3_hi);

    sum_hi = _mm_srli_epi32(sum_hi, 4);


    auto result = _mm_packus_epi32_sse2(sum_lo, sum_hi);

    sse_store_si128<uint16_t, true>(bufferp + x, result);
}

template <BorderMode border>
static inline void processBuffersBlock_sse(float* bufferp, const float* bufferLine, const int x)
{
    auto currLineM3 = sse_load_3_to_left_ps<float, border == BorderMode::LEFT, true>(bufferLine + x);
    auto currLineM2 = sse_load_2_to_left_ps<float, border == BorderMode::LEFT, true>(bufferLine + x);
    auto currLineM1 = sse_load_1_to_left_ps<float, border == BorderMode::LEFT, true>(bufferLine + x);
    auto currLine = sse_load_ps<float, true>(bufferLine + x);
    auto currLineP1 = sse_load_1_to_right_ps<float, false, true>(bufferLine + x);
    auto currLineP2 = sse_load_2_to_right_ps<float, false, true>(bufferLine + x);
    auto currLineP3 = sse_load_3_to_right_ps<float, false, true>(bufferLine + x);

    auto sum = _mm_mul_ps(currLineM3, currLineM2);
    sum = _mm_mul_ps(sum, currLineM1);
    sum = _mm_mul_ps(sum, currLine);
    sum = _mm_mul_ps(sum, currLineP1);
    sum = _mm_mul_ps(sum, currLineP2);
    sum = _mm_mul_ps(sum, currLineP3);

    const auto const_1_16 = _mm_set1_ps(1.0 / 16.0);

    auto result = _mm_mul_ps(sum, const_1_16);

    sse_store_ps<float, true>(bufferp + x, result);
}

static inline void processBuffers_sse(uint8_t* bufferp, int16_t* bufferLine, const int bufferStride, const int bufferHeight)
{
    auto bufferpp1 = bufferp;
    auto bufferpc0 = bufferpp1 + bufferStride;
    auto bufferpn1 = bufferpc0 + bufferStride;

    constexpr int pixelStep = sseBytes / sizeof(uint8_t);
    constexpr int pixelStep2 = sseBytes / sizeof(int16_t);

    for (int y = 0; y < bufferHeight - 1; ++y) {

        const auto const_0 = _mm_setzero_si128();

        for (int x = 0; x < bufferStride; x += pixelStep) {

            auto srcp1 = sse_load_si128<uint8_t, true>(bufferpp1 + x);
            auto srcc0 = sse_load_si128<uint8_t, true>(bufferpc0 + x);
            auto srcn1 = sse_load_si128<uint8_t, true>(bufferpn1 + x);

            auto srcp1_lo = _mm_unpacklo_epi8(srcp1, const_0);
            auto srcc0_lo = _mm_unpacklo_epi8(srcc0, const_0);
            auto srcn1_lo = _mm_unpacklo_epi8(srcn1, const_0);

            auto srcp1_hi = _mm_unpackhi_epi8(srcp1, const_0);
            auto srcc0_hi = _mm_unpackhi_epi8(srcc0, const_0);
            auto srcn1_hi = _mm_unpackhi_epi8(srcn1, const_0);

            auto sum_lo = _mm_add_epi16(srcp1_lo, srcc0_lo);
            auto sum_hi = _mm_add_epi16(srcp1_hi, srcc0_hi);

            sum_lo = _mm_add_epi16(sum_lo, srcn1_lo);
            sum_hi = _mm_add_epi16(sum_hi, srcn1_hi);

            sse_store_si128<int16_t, true>(bufferLine + x, sum_lo);
            sse_store_si128<int16_t, true>(bufferLine + x + pixelStep2, sum_hi);
        }

        processBuffersBlock_sse<BorderMode::LEFT>(bufferpc0, bufferLine, 0);

        for (int x = pixelStep; x < bufferStride - pixelStep; x += pixelStep)
            processBuffersBlock_sse<BorderMode::NONE>(bufferpc0, bufferLine, x);

        processBuffersBlock_sse<BorderMode::RIGHT>(bufferpc0, bufferLine, bufferStride - pixelStep);

        bufferpc0 += bufferStride;
        bufferpp1 += bufferStride;
        bufferpn1 += bufferStride;
    }
}

static inline void processBuffers_sse(uint16_t* bufferp, int32_t* bufferLine, const int bufferStride, const int bufferHeight)
{
    auto bufferpp1 = bufferp;
    auto bufferpc0 = bufferpp1 + bufferStride;
    auto bufferpn1 = bufferpc0 + bufferStride;

    constexpr int pixelStep = sseBytes / sizeof(uint16_t);
    constexpr int pixelStep2 = sseBytes / sizeof(int32_t);

    for (int y = 0; y < bufferHeight - 1; ++y) {

        const auto const_0 = _mm_setzero_si128();

        for (int x = 0; x < bufferStride; x += pixelStep) {

            auto srcp1 = sse_load_si128<uint16_t, true>(bufferpp1 + x);
            auto srcc0 = sse_load_si128<uint16_t, true>(bufferpc0 + x);
            auto srcn1 = sse_load_si128<uint16_t, true>(bufferpn1 + x);

            auto srcp1_lo = _mm_unpacklo_epi16(srcp1, const_0);
            auto srcc0_lo = _mm_unpacklo_epi16(srcc0, const_0);
            auto srcn1_lo = _mm_unpacklo_epi16(srcn1, const_0);

            auto srcp1_hi = _mm_unpackhi_epi16(srcp1, const_0);
            auto srcc0_hi = _mm_unpackhi_epi16(srcc0, const_0);
            auto srcn1_hi = _mm_unpackhi_epi16(srcn1, const_0);

            auto sum_lo = _mm_add_epi32(srcp1_lo, srcc0_lo);
            auto sum_hi = _mm_add_epi32(srcp1_hi, srcc0_hi);

            sum_lo = _mm_add_epi32(sum_lo, srcn1_lo);
            sum_hi = _mm_add_epi32(sum_hi, srcn1_hi);

            sse_store_si128<int32_t, true>(bufferLine + x, sum_lo);
            sse_store_si128<int32_t, true>(bufferLine + x + pixelStep2, sum_hi);
        }

        processBuffersBlock_sse<BorderMode::LEFT>(bufferpc0, bufferLine, 0);

        for (int x = pixelStep; x < bufferStride - pixelStep; x += pixelStep)
            processBuffersBlock_sse<BorderMode::NONE>(bufferpc0, bufferLine, x);

        processBuffersBlock_sse<BorderMode::RIGHT>(bufferpc0, bufferLine, bufferStride - pixelStep);


        bufferpc0 += bufferStride;
        bufferpp1 += bufferStride;
        bufferpn1 += bufferStride;
    }
}

static inline void processBuffers_sse(float* bufferp, float* bufferLine, const int bufferStride, const int bufferHeight)
{
    auto bufferpp1 = bufferp;
    auto bufferpc0 = bufferpp1 + bufferStride;
    auto bufferpn1 = bufferpc0 + bufferStride;

    constexpr int pixelStep = sseBytes / sizeof(float);

    for (int y = 0; y < bufferHeight - 1; ++y) {

        for (int x = 0; x < bufferStride; x += pixelStep) {

            auto srcp1 = sse_load_ps<float, true>(bufferpp1 + x);
            auto srcc0 = sse_load_ps<float, true>(bufferpc0 + x);
            auto srcn1 = sse_load_ps<float, true>(bufferpn1 + x);

            auto sum = _mm_add_ps(srcp1, srcc0);

            sum = _mm_add_ps(sum, srcn1);

            sse_store_ps<float, true>(bufferLine + x, sum);
        }

        processBuffersBlock_sse<BorderMode::LEFT>(bufferpc0, bufferLine, 0);

        for (int x = pixelStep; x < bufferStride - pixelStep; x += pixelStep)
            processBuffersBlock_sse<BorderMode::NONE>(bufferpc0, bufferLine, x);

        processBuffersBlock_sse<BorderMode::RIGHT>(bufferpc0, bufferLine, bufferStride - pixelStep);


        bufferpc0 += bufferStride;
        bufferpp1 += bufferStride;
        bufferpn1 += bufferStride;
    }
}

#endif

template <typename T, typename IType>
static inline void processBuffers_c(T* bufferp, IType* bufferLine, const int bufferStride, const int bufferHeight)
{
    auto bufferpc = bufferp + bufferStride;
    auto bufferpp1 = bufferpc - bufferStride;
    auto bufferpn1 = bufferpc + bufferStride;

    for (int y = 0; y < bufferHeight - 1; ++y) {

        for (int x = 0; x < bufferStride; ++x) {
            bufferLine[x] = bufferpp1[x] + bufferpc[x] + bufferpn1[x];
        }

        for (int x = 0; x < bufferStride; ++x) {

            const IType currLineM3 = loadPixel<IType, IType>(bufferLine, x, -3, bufferStride);
            const IType currLineM2 = loadPixel<IType, IType>(bufferLine, x, -2, bufferStride);
            const IType currLineM1 = loadPixel<IType, IType>(bufferLine, x, -1, bufferStride);
            const IType currLine = bufferLine[x];
            const IType currLineP1 = loadPixel<IType, IType>(bufferLine, x, 1, bufferStride);
            const IType currLineP2 = loadPixel<IType, IType>(bufferLine, x, 2, bufferStride);
            const IType currLineP3 = loadPixel<IType, IType>(bufferLine, x, 3, bufferStride);

            bufferpc[x] = (currLineM3 + currLineM2 + currLineM1 + currLine + currLineP1 + currLineP2 + currLineP3) / 16;
        }

        bufferpc += bufferStride;
        bufferpp1 += bufferStride;
        bufferpn1 += bufferStride;
    }
}

#ifdef VS_TARGET_CPU_X86

template <BorderMode border, bool alignedLoad, bool alignedLoadBuffer, bool alignedStore>
static inline void finalizePlaneLine_sse(const uint8_t* srcp, const uint8_t* srcpn2, uint8_t* dstpn, uint8_t* buffers[TOTAL_BUFFERS], int bufferOffset, const int w, const int pixelStep, const float aaf)
{
    const auto aa = _mm_set1_epi8(aaf);
    const auto const_0 = _mm_setzero_si128();

    for (int x = 0; x < w; x += pixelStep) {

        auto currLineM3 = sse_load_3_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM2 = sse_load_2_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM1 = sse_load_1_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLine = sse_load_si128<uint8_t, alignedLoad>(srcp + x);
        auto currLineP1 = sse_load_1_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP2 = sse_load_2_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP3 = sse_load_3_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);

        auto nextLineM3 = sse_load_3_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM2 = sse_load_2_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM1 = sse_load_1_to_left_si128<uint8_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLine = sse_load_si128<uint8_t, alignedLoad>(srcpn2 + x);
        auto nextLineP1 = sse_load_1_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP2 = sse_load_2_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP3 = sse_load_3_to_right_si128<uint8_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);

        auto forwardSangNom1 = calculateSangNom_sse<uint8_t, __m128i>(currLineM1, currLine, currLineP1);
        auto forwardSangNom2 = calculateSangNom_sse<uint8_t, __m128i>(nextLineP1, nextLine, nextLineM1);
        auto backwardSangNom1 = calculateSangNom_sse<uint8_t, __m128i>(currLineP1, currLine, currLineM1);
        auto backwardSangNom2 = calculateSangNom_sse<uint8_t, __m128i>(nextLineM1, nextLine, nextLineP1);

        auto buf0 = sse_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_M3_P3] + bufferOffset + x);
        auto buf1 = sse_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_M2_P2] + bufferOffset + x);
        auto buf2 = sse_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_M1_P1] + bufferOffset + x);
        auto buf3 = sse_load_si128<uint8_t, alignedLoadBuffer>(buffers[SG_FORWARD] + bufferOffset + x);
        auto buf4 = sse_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_P0_M0] + bufferOffset + x);
        auto buf5 = sse_load_si128<uint8_t, alignedLoadBuffer>(buffers[SG_REVERSE] + bufferOffset + x);
        auto buf6 = sse_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_P1_M1] + bufferOffset + x);
        auto buf7 = sse_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_P2_M2] + bufferOffset + x);
        auto buf8 = sse_load_si128<uint8_t, alignedLoadBuffer>(buffers[ADIFF_P3_M3] + bufferOffset + x);

        auto minBuf = _mm_min_epu8(buf0, buf1);
        minBuf = _mm_min_epu8(minBuf, buf2);
        minBuf = _mm_min_epu8(minBuf, buf3);
        minBuf = _mm_min_epu8(minBuf, buf4);
        minBuf = _mm_min_epu8(minBuf, buf5);
        minBuf = _mm_min_epu8(minBuf, buf6);
        minBuf = _mm_min_epu8(minBuf, buf7);
        minBuf = _mm_min_epu8(minBuf, buf8);

        auto minBufAvg = _mm_setzero_si128();

        ///////////////////////////////////////////////////////////////////////
        // the order of getAvgIfMinBuf is important, don't change them
        minBufAvg = getAvgIfMinBuf<uint8_t, __m128i>(currLineM3, nextLineP3, buf0, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<uint8_t, __m128i>(currLineP3, nextLineM3, buf8, minBuf, minBufAvg);

        minBufAvg = getAvgIfMinBuf<uint8_t, __m128i>(currLineM2, nextLineP2, buf1, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<uint8_t, __m128i>(currLineP2, nextLineM2, buf7, minBuf, minBufAvg);

        minBufAvg = getAvgIfMinBuf<uint8_t, __m128i>(currLineM1, nextLineP1, buf2, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<uint8_t, __m128i>(currLineP1, nextLineM1, buf6, minBuf, minBufAvg);

        minBufAvg = getAvgIfMinBuf<uint8_t, __m128i>(forwardSangNom1, forwardSangNom2, buf3, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<uint8_t, __m128i>(backwardSangNom1, backwardSangNom2, buf5, minBuf, minBufAvg);
        ///////////////////////////////////////////////////////////////////////

        auto buf4Avg = _mm_avg_epu8(currLine, nextLine);

        auto buf4IsMin = _mm_cmpeq_epi8(buf4, minBuf);

        auto takeAA = _mm_subs_epu8(minBuf, aa);
        auto takeMinBufAvg = _mm_cmpeq_epi8(takeAA, const_0);

        auto mask = _mm_andnot_si128(buf4IsMin, takeMinBufAvg);

        minBufAvg = _mm_and_si128(mask, minBufAvg);
        buf4Avg = _mm_andnot_si128(mask, buf4Avg);
        auto result = _mm_or_si128(minBufAvg, buf4Avg);

        sse_store_si128<uint8_t, alignedStore>(dstpn + x, result);
    }
}

template <BorderMode border, bool alignedLoad, bool alignedLoadBuffer, bool alignedStore>
static inline void finalizePlaneLine_sse(const uint16_t* srcp, const uint16_t* srcpn2, uint16_t* dstpn, uint16_t* buffers[TOTAL_BUFFERS], int bufferOffset, const int w, const int pixelStep, const float aaf)
{
    const auto aa = _mm_set1_epi16(aaf);
    const auto const_0 = _mm_setzero_si128();

    for (int x = 0; x < w; x += pixelStep) {

        auto currLineM3 = sse_load_3_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM2 = sse_load_2_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM1 = sse_load_1_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLine = sse_load_si128<uint16_t, alignedLoad>(srcp + x);
        auto currLineP1 = sse_load_1_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP2 = sse_load_2_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP3 = sse_load_3_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcp + x);

        auto nextLineM3 = sse_load_3_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM2 = sse_load_2_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM1 = sse_load_1_to_left_si128<uint16_t, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLine = sse_load_si128<uint16_t, alignedLoad>(srcpn2 + x);
        auto nextLineP1 = sse_load_1_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP2 = sse_load_2_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP3 = sse_load_3_to_right_si128<uint16_t, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);

        auto forwardSangNom1 = calculateSangNom_sse<uint16_t, __m128i>(currLineM1, currLine, currLineP1);
        auto forwardSangNom2 = calculateSangNom_sse<uint16_t, __m128i>(nextLineP1, nextLine, nextLineM1);
        auto backwardSangNom1 = calculateSangNom_sse<uint16_t, __m128i>(currLineP1, currLine, currLineM1);
        auto backwardSangNom2 = calculateSangNom_sse<uint16_t, __m128i>(nextLineM1, nextLine, nextLineP1);

        auto buf0 = sse_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_M3_P3] + bufferOffset + x);
        auto buf1 = sse_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_M2_P2] + bufferOffset + x);
        auto buf2 = sse_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_M1_P1] + bufferOffset + x);
        auto buf3 = sse_load_si128<uint16_t, alignedLoadBuffer>(buffers[SG_FORWARD] + bufferOffset + x);
        auto buf4 = sse_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_P0_M0] + bufferOffset + x);
        auto buf5 = sse_load_si128<uint16_t, alignedLoadBuffer>(buffers[SG_REVERSE] + bufferOffset + x);
        auto buf6 = sse_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_P1_M1] + bufferOffset + x);
        auto buf7 = sse_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_P2_M2] + bufferOffset + x);
        auto buf8 = sse_load_si128<uint16_t, alignedLoadBuffer>(buffers[ADIFF_P3_M3] + bufferOffset + x);

        auto minBuf = _mm_min_epi16(convSign16(buf0), convSign16(buf1));
        minBuf = _mm_min_epi16(minBuf, convSign16(buf2));
        minBuf = _mm_min_epi16(minBuf, convSign16(buf3));
        minBuf = _mm_min_epi16(minBuf, convSign16(buf4));
        minBuf = _mm_min_epi16(minBuf, convSign16(buf5));
        minBuf = _mm_min_epi16(minBuf, convSign16(buf6));
        minBuf = _mm_min_epi16(minBuf, convSign16(buf7));
        minBuf = _mm_min_epi16(minBuf, convSign16(buf8));
        minBuf = convSign16(minBuf);

        auto minBufAvg = _mm_setzero_si128();

        ///////////////////////////////////////////////////////////////////////
        // the order of getAvgIfMinBuf is important, don't change them
        minBufAvg = getAvgIfMinBuf<uint16_t, __m128i>(currLineM3, nextLineP3, buf0, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<uint16_t, __m128i>(currLineP3, nextLineM3, buf8, minBuf, minBufAvg);

        minBufAvg = getAvgIfMinBuf<uint16_t, __m128i>(currLineM2, nextLineP2, buf1, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<uint16_t, __m128i>(currLineP2, nextLineM2, buf7, minBuf, minBufAvg);

        minBufAvg = getAvgIfMinBuf<uint16_t, __m128i>(currLineM1, nextLineP1, buf2, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<uint16_t, __m128i>(currLineP1, nextLineM1, buf6, minBuf, minBufAvg);

        minBufAvg = getAvgIfMinBuf<uint16_t, __m128i>(forwardSangNom1, forwardSangNom2, buf3, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<uint16_t, __m128i>(backwardSangNom1, backwardSangNom2, buf5, minBuf, minBufAvg);
        ///////////////////////////////////////////////////////////////////////

        auto buf4Avg = _mm_avg_epu16(currLine, nextLine);

        auto buf4IsMin = _mm_cmpeq_epi16(buf4, minBuf);

        auto takeAA = _mm_subs_epu16(minBuf, aa);
        auto takeMinBufAvg = _mm_cmpeq_epi16(takeAA, const_0);

        auto mask = _mm_andnot_si128(buf4IsMin, takeMinBufAvg);

        minBufAvg = _mm_and_si128(mask, minBufAvg);
        buf4Avg = _mm_andnot_si128(mask, buf4Avg);
        auto result = _mm_or_si128(minBufAvg, buf4Avg);

        sse_store_si128<uint16_t, alignedStore>(dstpn + x, result);
    }
}

template <BorderMode border, bool alignedLoad, bool alignedLoadBuffer, bool alignedStore>
static inline void finalizePlaneLine_sse(const float* srcp, const float* srcpn2, float* dstpn, float* buffers[TOTAL_BUFFERS], int bufferOffset, const int w, const int pixelStep, const float aaf)
{
    const auto aa = _mm_set1_ps(aaf);
    const auto const_0 = _mm_setzero_ps();
    const auto const_1_2 = _mm_set1_ps(1.0 / 2.0);

    for (int x = 0; x < w; x += pixelStep) {

        auto currLineM3 = sse_load_3_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM2 = sse_load_2_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLineM1 = sse_load_1_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcp + x);
        auto currLine = sse_load_ps<float, alignedLoad>(srcp + x);
        auto currLineP1 = sse_load_1_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP2 = sse_load_2_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcp + x);
        auto currLineP3 = sse_load_3_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcp + x);

        auto nextLineM3 = sse_load_3_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM2 = sse_load_2_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLineM1 = sse_load_1_to_left_ps<float, border == BorderMode::LEFT, alignedLoad>(srcpn2 + x);
        auto nextLine = sse_load_ps<float, alignedLoad>(srcpn2 + x);
        auto nextLineP1 = sse_load_1_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP2 = sse_load_2_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);
        auto nextLineP3 = sse_load_3_to_right_ps<float, border == BorderMode::RIGHT, alignedLoad>(srcpn2 + x);

        auto forwardSangNom1 = calculateSangNom_sse<float, __m128>(currLineM1, currLine, currLineP1);
        auto forwardSangNom2 = calculateSangNom_sse<float, __m128>(nextLineP1, nextLine, nextLineM1);
        auto backwardSangNom1 = calculateSangNom_sse<float, __m128>(currLineP1, currLine, currLineM1);
        auto backwardSangNom2 = calculateSangNom_sse<float, __m128>(nextLineM1, nextLine, nextLineP1);

        auto buf0 = sse_load_ps<float, alignedLoadBuffer>(buffers[ADIFF_M3_P3] + bufferOffset + x);
        auto buf1 = sse_load_ps<float, alignedLoadBuffer>(buffers[ADIFF_M2_P2] + bufferOffset + x);
        auto buf2 = sse_load_ps<float, alignedLoadBuffer>(buffers[ADIFF_M1_P1] + bufferOffset + x);
        auto buf3 = sse_load_ps<float, alignedLoadBuffer>(buffers[SG_FORWARD] + bufferOffset + x);
        auto buf4 = sse_load_ps<float, alignedLoadBuffer>(buffers[ADIFF_P0_M0] + bufferOffset + x);
        auto buf5 = sse_load_ps<float, alignedLoadBuffer>(buffers[SG_REVERSE] + bufferOffset + x);
        auto buf6 = sse_load_ps<float, alignedLoadBuffer>(buffers[ADIFF_P1_M1] + bufferOffset + x);
        auto buf7 = sse_load_ps<float, alignedLoadBuffer>(buffers[ADIFF_P2_M2] + bufferOffset + x);
        auto buf8 = sse_load_ps<float, alignedLoadBuffer>(buffers[ADIFF_P3_M3] + bufferOffset + x);

        auto minBuf = _mm_min_ps(buf0, buf1);
        minBuf = _mm_min_ps(minBuf, buf2);
        minBuf = _mm_min_ps(minBuf, buf3);
        minBuf = _mm_min_ps(minBuf, buf4);
        minBuf = _mm_min_ps(minBuf, buf5);
        minBuf = _mm_min_ps(minBuf, buf6);
        minBuf = _mm_min_ps(minBuf, buf7);
        minBuf = _mm_min_ps(minBuf, buf8);

        auto minBufAvg = _mm_setzero_ps();

        ///////////////////////////////////////////////////////////////////////
        // the order of getAvgIfMinBuf is important, don't change them
        minBufAvg = getAvgIfMinBuf<float, __m128>(currLineM3, nextLineP3, buf0, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<float, __m128>(currLineP3, nextLineM3, buf8, minBuf, minBufAvg);

        minBufAvg = getAvgIfMinBuf<float, __m128>(currLineM2, nextLineP2, buf1, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<float, __m128>(currLineP2, nextLineM2, buf7, minBuf, minBufAvg);

        minBufAvg = getAvgIfMinBuf<float, __m128>(currLineM1, nextLineP1, buf2, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<float, __m128>(currLineP1, nextLineM1, buf6, minBuf, minBufAvg);

        minBufAvg = getAvgIfMinBuf<float, __m128>(forwardSangNom1, forwardSangNom2, buf3, minBuf, minBufAvg);
        minBufAvg = getAvgIfMinBuf<float, __m128>(backwardSangNom1, backwardSangNom2, buf5, minBuf, minBufAvg);
        ///////////////////////////////////////////////////////////////////////

        auto buf4Avg = _mm_mul_ps(_mm_add_ps(currLine, nextLine), const_1_2);

        auto buf4IsMin = _mm_cmpeq_ps(buf4, minBuf);

        auto takeAA = _mm_sub_ps(minBuf, aa);
        auto takeMinBufAvg = _mm_cmple_ps(takeAA, const_0);

        auto mask = _mm_andnot_ps(buf4IsMin, takeMinBufAvg);

        minBufAvg = _mm_and_ps(mask, minBufAvg);
        buf4Avg = _mm_andnot_ps(mask, buf4Avg);
        auto result = _mm_or_ps(minBufAvg, buf4Avg);

        sse_store_ps<float, alignedStore>(dstpn + x, result);
    }
}

template <typename T, typename IType>
static inline void finalizePlane_sse(T* dstp, const int dstStride, const int w, const int h, const int bufferStride, const T aaf, T* buffers[TOTAL_BUFFERS])
{
    auto srcp = dstp;
    auto srcpn2 = srcp + dstStride * static_cast<int64_t>(2);
    auto dstpn = dstp + dstStride;

    int bufferOffset = bufferStride;

    constexpr int pixelStep = sseBytes / sizeof(T);
    const int wMod = (w - 1) & ~(pixelStep - 1);

    for (int y = 0; y < h / 2 - 1; ++y) {

        finalizePlaneLine_sse<BorderMode::LEFT, true, true, true>(srcp, srcpn2, dstpn, buffers, bufferOffset, pixelStep, pixelStep, aaf);
        finalizePlaneLine_sse<BorderMode::NONE, true, true, true>(srcp + pixelStep, srcpn2 + pixelStep, dstpn + pixelStep, buffers, bufferOffset + pixelStep, wMod - pixelStep, pixelStep, aaf);
        finalizePlaneLine_sse<BorderMode::RIGHT, false, false, false>(srcp + w - pixelStep, srcpn2 + w - pixelStep, dstpn + w - pixelStep, buffers, bufferOffset + w - pixelStep, pixelStep, pixelStep, aaf);

        srcp += dstStride * static_cast<int64_t>(2);
        srcpn2 += dstStride * static_cast<int64_t>(2);
        dstpn += dstStride * static_cast<int64_t>(2);
        bufferOffset += bufferStride;
    }
}

#endif

template <typename T, typename IType>
static inline void finalizePlane_c(T* dstp, const int dstStride, const int w, const int h, const int bufferStride, const T aaf, T* buffers[TOTAL_BUFFERS])
{
    auto srcp = dstp;
    auto srcpn2 = srcp + dstStride * 2;
    auto dstpn = dstp + dstStride;

    int bufferOffset = bufferStride;

    for (int y = 0; y < h / 2 - 1; ++y) {

        for (int x = 0; x < w; ++x) {

            const IType currLineM3 = loadPixel<T, IType>(srcp, x, -3, w);
            const IType currLineM2 = loadPixel<T, IType>(srcp, x, -2, w);
            const IType currLineM1 = loadPixel<T, IType>(srcp, x, -1, w);
            const IType currLine = srcp[x];
            const IType currLineP1 = loadPixel<T, IType>(srcp, x, 1, w);
            const IType currLineP2 = loadPixel<T, IType>(srcp, x, 2, w);
            const IType currLineP3 = loadPixel<T, IType>(srcp, x, 3, w);

            const IType nextLineM3 = loadPixel<T, IType>(srcpn2, x, -3, w);
            const IType nextLineM2 = loadPixel<T, IType>(srcpn2, x, -2, w);
            const IType nextLineM1 = loadPixel<T, IType>(srcpn2, x, -1, w);
            const IType nextLine = srcpn2[x];
            const IType nextLineP1 = loadPixel<T, IType>(srcpn2, x, 1, w);
            const IType nextLineP2 = loadPixel<T, IType>(srcpn2, x, 2, w);
            const IType nextLineP3 = loadPixel<T, IType>(srcpn2, x, 3, w);

            const IType forwardSangNom1 = calculateSangNom<T, IType>(currLineM1, currLine, currLineP1);
            const IType forwardSangNom2 = calculateSangNom<T, IType>(nextLineP1, nextLine, nextLineM1);
            const IType backwardSangNom1 = calculateSangNom<T, IType>(currLineP1, currLine, currLineM1);
            const IType backwardSangNom2 = calculateSangNom<T, IType>(nextLineM1, nextLine, nextLineP1);

            IType buf[9];
            buf[0] = buffers[ADIFF_M3_P3][bufferOffset + x];
            buf[1] = buffers[ADIFF_M2_P2][bufferOffset + x];
            buf[2] = buffers[ADIFF_M1_P1][bufferOffset + x];
            buf[3] = buffers[SG_FORWARD][bufferOffset + x];
            buf[4] = buffers[ADIFF_P0_M0][bufferOffset + x];
            buf[5] = buffers[SG_REVERSE][bufferOffset + x];
            buf[6] = buffers[ADIFF_P1_M1][bufferOffset + x];
            buf[7] = buffers[ADIFF_P2_M2][bufferOffset + x];
            buf[8] = buffers[ADIFF_P3_M3][bufferOffset + x];

            IType minBuf = buf[0];
            for (int i = 1; i < 9; ++i)
                minBuf = std::min(minBuf, buf[i]);

            ///////////////////////////////////////////////////////////////////////
            // the order of following code is important, don't change them
            if (buf[4] == minBuf || minBuf >= aaf) {
                dstpn[x] = (currLine + nextLine + 1) >> 1;
            }
            else if (buf[5] == minBuf) {
                dstpn[x] = (backwardSangNom1 + backwardSangNom2 + 1) >> 1;
            }
            else if (buf[3] == minBuf) {
                dstpn[x] = (forwardSangNom1 + forwardSangNom2 + 1) >> 1;
            }
            else if (buf[6] == minBuf) {
                dstpn[x] = (currLineP1 + nextLineM1 + 1) >> 1;
            }
            else if (buf[2] == minBuf) {
                dstpn[x] = (currLineM1 + nextLineP1 + 1) >> 1;
            }
            else if (buf[7] == minBuf) {
                dstpn[x] = (currLineP2 + nextLineM2 + 1) >> 1;
            }
            else if (buf[1] == minBuf) {
                dstpn[x] = (currLineM2 + nextLineP2 + 1) >> 1;
            }
            else if (buf[8] == minBuf) {
                dstpn[x] = (currLineP3 + nextLineM3 + 1) >> 1;
            }
            else if (buf[0] == minBuf) {
                dstpn[x] = (currLineM3 + nextLineP3 + 1) >> 1;
            }
        }

        srcp += dstStride * 2;
        srcpn2 += dstStride * 2;
        dstpn += dstStride * 2;
        bufferOffset += bufferStride;
    }
}

template <>
inline void finalizePlane_c<float, float>(float* dstp, const int dstStride, const int w, const int h, const int bufferStride, const float aaf, float* buffers[TOTAL_BUFFERS])
{
    auto srcp = dstp;
    auto srcpn2 = srcp + dstStride * static_cast<int64_t>(2);
    auto dstpn = dstp + dstStride;

    int bufferOffset = bufferStride;

    for (int y = 0; y < h / 2 - 1; ++y) {

        for (int x = 0; x < w; ++x) {

            const float currLineM3 = loadPixel<float, float>(srcp, x, -3, w);
            const float currLineM2 = loadPixel<float, float>(srcp, x, -2, w);
            const float currLineM1 = loadPixel<float, float>(srcp, x, -1, w);
            const float currLine = srcp[x];
            const float currLineP1 = loadPixel<float, float>(srcp, x, 1, w);
            const float currLineP2 = loadPixel<float, float>(srcp, x, 2, w);
            const float currLineP3 = loadPixel<float, float>(srcp, x, 3, w);

            const float nextLineM3 = loadPixel<float, float>(srcpn2, x, -3, w);
            const float nextLineM2 = loadPixel<float, float>(srcpn2, x, -2, w);
            const float nextLineM1 = loadPixel<float, float>(srcpn2, x, -1, w);
            const float nextLine = srcpn2[x];
            const float nextLineP1 = loadPixel<float, float>(srcpn2, x, 1, w);
            const float nextLineP2 = loadPixel<float, float>(srcpn2, x, 2, w);
            const float nextLineP3 = loadPixel<float, float>(srcpn2, x, 3, w);

            const float forwardSangNom1 = calculateSangNom<float, float>(currLineM1, currLine, currLineP1);
            const float forwardSangNom2 = calculateSangNom<float, float>(nextLineP1, nextLine, nextLineM1);
            const float backwardSangNom1 = calculateSangNom<float, float>(currLineP1, currLine, currLineM1);
            const float backwardSangNom2 = calculateSangNom<float, float>(nextLineM1, nextLine, nextLineP1);

            float buf[9];
            buf[0] = buffers[ADIFF_M3_P3][bufferOffset + x];
            buf[1] = buffers[ADIFF_M2_P2][bufferOffset + x];
            buf[2] = buffers[ADIFF_M1_P1][bufferOffset + x];
            buf[3] = buffers[SG_FORWARD][bufferOffset + x];
            buf[4] = buffers[ADIFF_P0_M0][bufferOffset + x];
            buf[5] = buffers[SG_REVERSE][bufferOffset + x];
            buf[6] = buffers[ADIFF_P1_M1][bufferOffset + x];
            buf[7] = buffers[ADIFF_P2_M2][bufferOffset + x];
            buf[8] = buffers[ADIFF_P3_M3][bufferOffset + x];

            float minBuf = buf[0];
            for (int i = 1; i < 9; ++i)
                minBuf = std::min(minBuf, buf[i]);

            ///////////////////////////////////////////////////////////////////////
            // the order of following code is important, don't change them
            if (buf[4] == minBuf || minBuf >= aaf) {
                dstpn[x] = (currLine + nextLine) / 2;
            }
            else if (buf[5] == minBuf) {
                dstpn[x] = (backwardSangNom1 + backwardSangNom2) / 2;
            }
            else if (buf[3] == minBuf) {
                dstpn[x] = (forwardSangNom1 + forwardSangNom2) / 2;
            }
            else if (buf[6] == minBuf) {
                dstpn[x] = (currLineP1 + nextLineM1) / 2;
            }
            else if (buf[2] == minBuf) {
                dstpn[x] = (currLineM1 + nextLineP1) / 2;
            }
            else if (buf[7] == minBuf) {
                dstpn[x] = (currLineP2 + nextLineM2) / 2;
            }
            else if (buf[1] == minBuf) {
                dstpn[x] = (currLineM2 + nextLineP2) / 2;
            }
            else if (buf[8] == minBuf) {
                dstpn[x] = (currLineP3 + nextLineM3) / 2;
            }
            else if (buf[0] == minBuf) {
                dstpn[x] = (currLineM3 + nextLineP3) / 2;
            }
        }

        srcp += dstStride * static_cast<int64_t>(2);
        srcpn2 += dstStride * static_cast<int64_t>(2);
        dstpn += dstStride * static_cast<int64_t>(2);
        bufferOffset += bufferStride;
    }
}


#ifdef VS_TARGET_CPU_X86

template <typename T, typename IType>
void SangNom2::sangnom_sse(T* dstp, const int dstStride, const int w, const int h, int offset, int plane, T* buffers[TOTAL_BUFFERS], IType* bufferLine)
{
    prepareBuffers_sse<T, IType>(dstp + static_cast<int64_t>(offset) * dstStride, dstStride, w, h, bufferStride, buffers);

    for (int i = 0; i < TOTAL_BUFFERS; ++i)
        processBuffers_sse(buffers[i], bufferLine, bufferStride, bufferHeight);

    finalizePlane_sse<T, IType>(dstp + static_cast<int64_t>(offset) * dstStride, dstStride, w, h, bufferStride, aaf[plane], buffers);
}

#endif

template <typename T, typename IType>
void SangNom2::sangnom_c(T* dstp, const int dstStride, const int w, const int h, int offset, int plane, T* buffers[TOTAL_BUFFERS], IType* bufferLine)
{
    prepareBuffers_c<T, IType>(dstp + static_cast<int64_t>(offset) * dstStride, dstStride, w, h, bufferStride, buffers);

    for (int i = 0; i < TOTAL_BUFFERS; ++i)
        processBuffers_c<T, IType>(buffers[i], bufferLine, bufferStride, bufferHeight);

    finalizePlane_c<T, IType>(dstp + static_cast<int64_t>(offset) * dstStride, dstStride, w, h, bufferStride, aaf[plane], buffers);
}

SangNom2::SangNom2(PClip _child, int order, int aa, int aac, int threads, bool dh, bool luma, bool chroma, IScriptEnvironment* env)
    : GenericVideoFilter(_child), _order(order), _aa(aa), _aac(aac), _dh(dh), _luma(luma), _chroma(chroma)
{
    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    if (vi.IsRGB())
        env->ThrowError("SangNom2: clip must be in Y/YUV format.");

    if (vi.height % 2 != 0)
        env->ThrowError("SangNom2: height must be even.");

    if (_order < 0 || _order > 2)
        env->ThrowError("SangNom2: order must be between 0..2.");

    if (_aa < 0 || _aa > 128)
        env->ThrowError("SangNom2: aa must be between 0..128.");

    if (_aac < 0 || _aac > 128)
        env->ThrowError("SangNom2: aac must be between 0..128.");

    int planecount = min(vi.NumComponents(), 3);
    for (int i = 0; i < planecount; i++)
    {
        if (i == 0)
        {
            processPlane[i] = _luma;
            if (vi.BitsPerComponent() != 32)
                aaf[i] = (_aa * 21.0f / 16.0f) * (1 << (vi.BitsPerComponent() - 8));
            else
                aaf[i] = (_aa * 21.0f / 16.0f) / 256.0f;
        }
        else
        {
            processPlane[i] = _chroma;
            if (vi.BitsPerComponent() != 32)
                aaf[i] = (_aac * 21.0f / 16.0f) * (1 << (vi.BitsPerComponent() - 8));
            else
                aaf[i] = (_aac * 21.0f / 16.0f) / 256.0f;
        }
    }

    if (_dh)
        vi.height *= 2;

    bufferStride = (vi.width + alignment - 1) & ~(alignment - 1);
    bufferHeight = (vi.height + 1) >> 1;
}

PVideoFrame SangNom2::GetFrame(int n, IScriptEnvironment* env)
{
    auto src = child->GetFrame(n, env);

    int offset = 42; // Initialise it to shut up a warning.

    if (_order == SNOT_DFR)
    {
        if (child->GetParity(n))
            offset = 0; // top field first
        else
            offset = 1; // bottom field first
    }
    else if (_order == SNOT_SFR_KT)
        offset = 0;
    else
        offset = 1;

    //auto dst = vsapi->copyFrame(src, core);
    auto dst = env->NewVideoFrame(vi);
    if (has_at_least_v8)
        dst = env->NewVideoFrameP(vi, &src, alignment);
    else
        dst = env->NewVideoFrame(vi);

    /////////////////////////////////////////////////////////////////////////////////////
    void* bufferLine;               // line buffer used in process buffers

    void* bufferPool;
    void* buffers[TOTAL_BUFFERS];   // plane buffers used in all three steps

    if (has_at_least_v8)
    {
        if (vi.BitsPerComponent() != 32)
        {
            if (vi.BitsPerComponent() == 8)
            {
                bufferLine = env->Allocate(sizeof(int16_t) * bufferStride, alignment, AVS_POOLED_ALLOC);
                bufferPool = env->Allocate(sizeof(uint8_t) * bufferStride * (bufferHeight + static_cast<int64_t>(1)) * TOTAL_BUFFERS, alignment, AVS_POOLED_ALLOC);
                // separate bufferpool to multiple pieces
                for (int i = 0; i < TOTAL_BUFFERS; ++i)
                    buffers[i] = reinterpret_cast<uint8_t*>(bufferPool) + static_cast<int64_t>(i) * bufferStride * (bufferHeight + static_cast<int64_t>(1));
            }
            else
            {
                bufferLine = env->Allocate(sizeof(int32_t) * bufferStride, alignment, AVS_POOLED_ALLOC);
                bufferPool = env->Allocate(sizeof(uint16_t) * bufferStride * (bufferHeight + static_cast<int64_t>(1)) * TOTAL_BUFFERS, alignment, AVS_POOLED_ALLOC);
                // separate bufferpool to multiple pieces
                for (int i = 0; i < TOTAL_BUFFERS; ++i)
                    buffers[i] = reinterpret_cast<uint16_t*>(bufferPool) + static_cast<int64_t>(i) * bufferStride * (bufferHeight + static_cast<int64_t>(1));
            }
        }
        else
        {
            bufferLine = env->Allocate(sizeof(float) * bufferStride, alignment, AVS_POOLED_ALLOC);
            bufferPool = env->Allocate(sizeof(float) * bufferStride * (bufferHeight + static_cast<int64_t>(1)) * TOTAL_BUFFERS, alignment, AVS_POOLED_ALLOC);
            // separate bufferpool to multiple pieces
            for (int i = 0; i < TOTAL_BUFFERS; ++i)
                buffers[i] = reinterpret_cast<float*>(bufferPool) + static_cast<int64_t>(i) * bufferStride * (bufferHeight + static_cast<int64_t>(1));
        }
    }
    else
    {
        if (vi.BitsPerComponent() != 32)
        {
            if (vi.BitsPerComponent() == 8)
            {
                bufferLine = _aligned_malloc(sizeof(int16_t) * bufferStride, alignment);
                bufferPool = _aligned_malloc(sizeof(uint8_t) * bufferStride * (bufferHeight + static_cast<int64_t>(1)) * TOTAL_BUFFERS, alignment);
                // separate bufferpool to multiple pieces
                for (int i = 0; i < TOTAL_BUFFERS; ++i)
                    buffers[i] = reinterpret_cast<uint8_t*>(bufferPool) + static_cast<int64_t>(i) * bufferStride * (bufferHeight + static_cast<int64_t>(1));
            }
            else
            {
                bufferLine = _aligned_malloc(sizeof(int32_t) * bufferStride, alignment);
                bufferPool = _aligned_malloc(sizeof(uint16_t) * bufferStride * (bufferHeight + static_cast<int64_t>(1)) * TOTAL_BUFFERS, alignment);
                // separate bufferpool to multiple pieces
                for (int i = 0; i < TOTAL_BUFFERS; ++i)
                    buffers[i] = reinterpret_cast<uint16_t*>(bufferPool) + static_cast<int64_t>(i) * bufferStride * (bufferHeight + static_cast<int64_t>(1));
            }
        }
        else
        {
            bufferLine = _aligned_malloc(sizeof(float) * bufferStride, alignment);
            bufferPool = _aligned_malloc(sizeof(float) * bufferStride * (bufferHeight + static_cast<int64_t>(1)) * TOTAL_BUFFERS, alignment);
            // separate bufferpool to multiple pieces
            for (int i = 0; i < TOTAL_BUFFERS; ++i)
                buffers[i] = reinterpret_cast<float*>(bufferPool) + static_cast<int64_t>(i) * bufferStride * (bufferHeight + static_cast<int64_t>(1));
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int* current_planes = planes_y;
    int planecount = min(vi.NumComponents(), 3);
    for (int i = 0; i < planecount; i++)
    {
        const int plane = current_planes[i];

        auto srcp = src->GetReadPtr(plane);
        auto src_stride = src->GetPitch(plane);
        auto dst_stride = dst->GetPitch(plane);
        auto dstStride = dst->GetPitch(plane) / vi.ComponentSize();
        auto src_height = src->GetHeight(plane);
        auto dst_height = dst->GetHeight(plane);
        auto width = src->GetRowSize(plane);
        auto dstp = dst->GetWritePtr(plane);

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

#ifdef VS_TARGET_CPU_X86

        if (vi.BitsPerComponent() != 32)
        {
            if (vi.BitsPerComponent() == 8)
                sangnom_sse<uint8_t, int16_t>(dstp, dstStride, width, dst_height, offset, i, reinterpret_cast<uint8_t**>(buffers), reinterpret_cast<int16_t*>(bufferLine));
            else
                sangnom_sse<uint16_t, int32_t>(reinterpret_cast<uint16_t*>(dstp), dstStride, width / vi.ComponentSize(), dst_height, offset, i, reinterpret_cast<uint16_t**>(buffers), reinterpret_cast<int32_t*>(bufferLine));
        }
        else
        {
            //sangnom_sse<float, float>(reinterpret_cast<float*>(dstp), dstStride, width / vi.ComponentSize(), dst_height, offset, i, reinterpret_cast<float**>(buffers), reinterpret_cast<float*>(bufferLine));
            sangnom_c<float, float>(reinterpret_cast<float*>(dstp), dstStride, width / vi.ComponentSize(), dst_height, offset, i, reinterpret_cast<float**>(buffers), reinterpret_cast<float*>(bufferLine));
        }
#else

        if (vi.BitsPerComponent() != 32) {
            if (vi.BitsPerComponent() == 8)
                sangnom_c<uint8_t, int16_t>(dstp, dstStride, width, dst_height, offset, i, reinterpret_cast<uint8_t**>(buffers), reinterpret_cast<int16_t*>(bufferLine));
            else
                sangnom_c<uint16_t, int32_t>(reinterpret_cast<uint16_t*>(dstp), dstStride, width / vi.ComponentSize(), dst_height, offset, i, reinterpret_cast<uint16_t**>(buffers), reinterpret_cast<int32_t*>(bufferLine));
        }
        else
        {
            sangnom_c<float, float>(reinterpret_cast<float*>(dstp), dstStride, width / vi.ComponentSize(), dst_height, offset, i, reinterpret_cast<float**>(buffers), reinterpret_cast<float*>(bufferLine));
        }

#endif
    }

    if (has_at_least_v8)
    {
        env->Free(bufferLine);
        env->Free(bufferPool);
    }
    else
    {
        _aligned_free(bufferLine);
        _aligned_free(bufferPool);
    }

    return dst;
}

AVSValue __cdecl Create_SangNom2(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    return new SangNom2(
        args[0].AsClip(),
        args[1].AsInt(1),
        args[2].AsInt(48),
        args[3].AsInt(0),
        args[4].AsInt(0),
        args[5].AsBool(false),
        args[6].AsBool(true),
        args[7].AsBool(true),   
        env);
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("SangNom2", "c[order]i[aa]i[aac]i[threads]i[dh]b[luma]b[chroma]b", Create_SangNom2, 0);
    return "SangNom2";
}
