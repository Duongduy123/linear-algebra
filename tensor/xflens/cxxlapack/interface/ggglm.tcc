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

#ifndef CXXLAPACK_INTERFACE_GGGLM_TCC
#define CXXLAPACK_INTERFACE_GGGLM_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
ggglm(IndexType             n,
      IndexType             m,
      IndexType             p,
      float                 *A,
      IndexType             ldA,
      float                 *B,
      IndexType             ldB,
      float                 *d,
      float                 *x,
      float                 *y,
      float                 *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("sggglm");

    IndexType info;
    LAPACK_IMPL(sggglm)(&n,
                        &m,
                        &p,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        d,
                        x,
                        y,
                        work,
                        &lWork,
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
ggglm(IndexType             n,
      IndexType             m,
      IndexType             p,
      double                *A,
      IndexType             ldA,
      double                *B,
      IndexType             ldB,
      double                *d,
      double                *x,
      double                *y,
      double                *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("dggglm");

    IndexType info;
    LAPACK_IMPL(dggglm)(&n,
                        &m,
                        &p,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        d,
                        x,
                        y,
                        work,
                        &lWork,
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
ggglm(IndexType             n,
      IndexType             m,
      IndexType             p,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *B,
      IndexType             ldB,
      std::complex<float >  *d,
      std::complex<float >  *x,
      std::complex<float >  *y,
      std::complex<float >  *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("cggglm");

    IndexType info;
    LAPACK_IMPL(cggglm)(&n,
                        &m,
                        &p,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(B),
                        &ldB,
                        reinterpret_cast<float  *>(d),
                        reinterpret_cast<float  *>(x),
                        reinterpret_cast<float  *>(y),
                        reinterpret_cast<float  *>(work),
                        &lWork,
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
ggglm(IndexType             n,
      IndexType             m,
      IndexType             p,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *B,
      IndexType             ldB,
      std::complex<double>  *d,
      std::complex<double>  *x,
      std::complex<double>  *y,
      std::complex<double>  *work,
      IndexType             lWork)
{
    CXXLAPACK_DEBUG_OUT("zggglm");

    IndexType info;
    LAPACK_IMPL(zggglm)(&n,
                        &m,
                        &p,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(B),
                        &ldB,
                        reinterpret_cast<double *>(d),
                        reinterpret_cast<double *>(x),
                        reinterpret_cast<double *>(y),
                        reinterpret_cast<double *>(work),
                        &lWork,
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

#endif // CXXLAPACK_INTERFACE_GGGLM_TCC
