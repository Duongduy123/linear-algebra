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

#ifndef CXXLAPACK_INTERFACE_TGSY2_TCC
#define CXXLAPACK_INTERFACE_TGSY2_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
tgsy2(char                  trans,
      IndexType             ijob,
      IndexType             m,
      IndexType             n,
      const float           *A,
      IndexType             ldA,
      const float           *B,
      IndexType             ldB,
      float                 *C,
      IndexType             ldC,
      const float           *D,
      IndexType             ldD,
      const float           *E,
      IndexType             ldE,
      float                 *F,
      IndexType             ldF,
      float                 &scale,
      float                 &rdsum,
      float                 &rdscal,
      IndexType             *iWork,
      IndexType             &pq)

{
    CXXLAPACK_DEBUG_OUT("stgsy2");

    IndexType info;
    LAPACK_IMPL(stgsy2)(&trans,
                        &ijob,
                        &m,
                        &n,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        C,
                        &ldC,
                        D,
                        &ldD,
                        E,
                        &ldE,
                        F,
                        &ldF,
                        &scale,
                        &rdsum,
                        &rdscal,
                        iWork,
                        &pq,
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
tgsy2(char                  trans,
      IndexType             ijob,
      IndexType             m,
      IndexType             n,
      const double          *A,
      IndexType             ldA,
      const double          *B,
      IndexType             ldB,
      double                *C,
      IndexType             ldC,
      const double          *D,
      IndexType             ldD,
      const double          *E,
      IndexType             ldE,
      double                *F,
      IndexType             ldF,
      double                &scale,
      double                &rdsum,
      double                &rdscal,
      IndexType             *iWork,
      IndexType             &pq)

{
    CXXLAPACK_DEBUG_OUT("dtgsy2");

    IndexType info;
    LAPACK_IMPL(dtgsy2)(&trans,
                        &ijob,
                        &m,
                        &n,
                        A,
                        &ldA,
                        B,
                        &ldB,
                        C,
                        &ldC,
                        D,
                        &ldD,
                        E,
                        &ldE,
                        F,
                        &ldF,
                        &scale,
                        &rdsum,
                        &rdscal,
                        iWork,
                        &pq,
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
tgsy2(char                        trans,
      IndexType                   ijob,
      IndexType                   m,
      IndexType                   n,
      const std::complex<float >  *A,
      IndexType                   ldA,
      const std::complex<float >  *B,
      IndexType                   ldB,
      std::complex<float >        *C,
      IndexType                   ldC,
      const std::complex<float >  *D,
      IndexType                   ldD,
      const std::complex<float >  *E,
      IndexType                   ldE,
      std::complex<float >        *F,
      IndexType                   ldF,
      float                       &scale,
      float                       &rdsum,
      float                       &rdscal)

{
    CXXLAPACK_DEBUG_OUT("ctgsy2");

    IndexType info;
    LAPACK_IMPL(ctgsy2)(&trans,
                        &ijob,
                        &m,
                        &n,
                        reinterpret_cast<const float  *>(A),
                        &ldA,
                        reinterpret_cast<const float  *>(B),
                        &ldB,
                        reinterpret_cast<float  *>(C),
                        &ldC,
                        reinterpret_cast<const float  *>(D),
                        &ldD,
                        reinterpret_cast<const float  *>(E),
                        &ldE,
                        reinterpret_cast<float  *>(F),
                        &ldF,
                        &scale,
                        &rdsum,
                        &rdscal,
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
tgsy2(char                        trans,
      IndexType                   ijob,
      IndexType                   m,
      IndexType                   n,
      const std::complex<double>  *A,
      IndexType                   ldA,
      const std::complex<double>  *B,
      IndexType                   ldB,
      std::complex<double>        *C,
      IndexType                   ldC,
      const std::complex<double>  *D,
      IndexType                   ldD,
      const std::complex<double>  *E,
      IndexType                   ldE,
      std::complex<double>        *F,
      IndexType                   ldF,
      double                      &scale,
      double                      &rdsum,
      double                      &rdscal)
{
    CXXLAPACK_DEBUG_OUT("ztgsy2");

    IndexType info;
    LAPACK_IMPL(ztgsy2)(&trans,
                        &ijob,
                        &m,
                        &n,
                        reinterpret_cast<const double *>(A),
                        &ldA,
                        reinterpret_cast<const double *>(B),
                        &ldB,
                        reinterpret_cast<double *>(C),
                        &ldC,
                        reinterpret_cast<const double *>(D),
                        &ldD,
                        reinterpret_cast<const double *>(E),
                        &ldE,
                        reinterpret_cast<double *>(F),
                        &ldF,
                        &scale,
                        &rdsum,
                        &rdscal,
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

#endif // CXXLAPACK_INTERFACE_TGSY2_TCC
