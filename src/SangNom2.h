#pragma once

#include "avisynth.h"
#include "avs/minmax.h"

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

class SangNom2 : public GenericVideoFilter
{
    int _order;
    int _aa, _aac;
    bool _dh;
    bool _luma, _chroma;
    int opt_;
    float aaf[3];
    int bufferStride;
    int bufferHeight;
    bool processPlane[3];
    bool has_at_least_v8;
    void* bufferLine;
    void* bufferPool;
    IScriptEnvironment* env_;

    template <typename T, typename IType>
    void sangnom_sse(T* dstp, const int dstStride, const int w, const int h, int offset, int plane, T* buffers[TOTAL_BUFFERS], IType* bufferLine);
    template <typename T, typename IType>
    void sangnom_c(T* dstp, const int dstStride, const int w, const int h, int offset, int plane, T* buffers[TOTAL_BUFFERS], IType* bufferLine);


public:
    SangNom2(PClip _child, int order, int aa, int aac, int threads, bool dh, bool luma, bool chroma, int opt, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    }
    ~SangNom2();
};
