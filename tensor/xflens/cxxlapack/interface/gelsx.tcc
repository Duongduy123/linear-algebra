/*
 *   Copyright (c) 2012, Michael Lehn, Klaus Pototzky
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CXXLAPACK_INTERFACE_GELSX_TCC
#define CXXLAPACK_INTERFACE_GELSX_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
gelsx(IndexType             m,
      IndexType             n,
      IndexType             nRhs,
      float                 *A,
      IndexType             ldA,
      float                 *B,
      IndexType             ldB,
      IndexType             *jpvt,
      float                 rCond,
      IndexType             &rank,
      float                 *work)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("sgelsx");
    LAPACK_IMPL(sgelsx)(&m,
                        &n,
                        &nRhs,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        jpvt,
                        &rCond,
                        &rank,
                        work,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
gelsx(IndexType             m,
      IndexType             n,
      IndexType             nRhs,
      double                *A,
      IndexType             ldA,
      double                *B,
      IndexType             ldB,
      IndexType             *jpvt,
      double                rCond,
      IndexType             &rank,
      double                *work)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("dgelsx");
    LAPACK_IMPL(dgelsx)(&m,
                        &n,
                        &nRhs,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        jpvt,
                        &rCond,
                        &rank,
                        work,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
gelsx(IndexType             m,
      IndexType             n,
      IndexType             nRhs,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *B,
      IndexType             ldB,
      IndexType             *jpvt,
      float                 rCond,
      IndexType             &rank,
      std::complex<float >  *work,
      float                 *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("cgelsx");
    LAPACK_IMPL(cgelsx)(&m,
                        &n,
                        &nRhs,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        jpvt,
                        &rCond,
                        &rank,
                        reinterpret_cast<float  *>(work),
                        rWork,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
gelsx(IndexType             m,
      IndexType             n,
      IndexType             nRhs,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *B,
      IndexType             ldB,
      IndexType             *jpvt,
      double                rCond,
      IndexType             &rank,
      std::complex<double>  *work,
      double                *rWork)
{
    IndexType info;
    CXXLAPACK_DEBUG_OUT("zgelsx");
    LAPACK_IMPL(zgelsx)(&m,
                        &n,
                        &nRhs,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        jpvt,
                        &rCond,
                        &rank,
                        reinterpret_cast<double *>(work),
                        rWork,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_GELSX_TCC
