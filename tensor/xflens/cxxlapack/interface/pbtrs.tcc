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

#ifndef CXXLAPACK_INTERFACE_PBTRS_TCC
#define CXXLAPACK_INTERFACE_PBTRS_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
pbtrs(char                  uplo,
      IndexType             n,
      IndexType             kd,
      IndexType             nRhs,
      const float           *Ab,
      IndexType             ldAb,
      float                 *B,
      IndexType             ldB)
{
    CXXLAPACK_DEBUG_OUT("spbtrs");

    IndexType info;
    LAPACK_IMPL(spbtrs)(&uplo,
                        &n,
                        &kd,
                        &nRhs,
                        Ab,
                        &ldAb,
                        B,
                        &ldB,
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
pbtrs(char                  uplo,
      IndexType             n,
      IndexType             kd,
      IndexType             nRhs,
      const double          *Ab,
      IndexType             ldAb,
      double                *B,
      IndexType             ldB)
{
    CXXLAPACK_DEBUG_OUT("dpbtrs");

    IndexType info;
    LAPACK_IMPL(dpbtrs)(&uplo,
                        &n,
                        &kd,
                        &nRhs,
                        Ab,
                        &ldAb,
                        B,
                        &ldB,
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
pbtrs(char                        uplo,
      IndexType                   n,
      IndexType                   kd,
      IndexType                   nRhs,
      const std::complex<float >  *Ab,
      IndexType                    ldAb,
      std::complex<float >         *B,
      IndexType                    ldB)
{
    CXXLAPACK_DEBUG_OUT("cpbtrs");

    IndexType info;
    LAPACK_IMPL(cpbtrs)(&uplo,
                        &n,
                        &kd,
                        &nRhs,
                        reinterpret_cast<const float  *>(Ab),
                        &ldAb,
                        reinterpret_cast<float  *>(B),
                        &ldB,
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
pbtrs(char                        uplo,
      IndexType                   n,
      IndexType                   kd,
      IndexType                   nRhs,
      const std::complex<double>  *Ab,
      IndexType                    ldAb,
      std::complex<double>         *B,
      IndexType                    ldB)
{
    CXXLAPACK_DEBUG_OUT("zpbtrs");

    IndexType info;
    LAPACK_IMPL(zpbtrs)(&uplo,
                        &n,
                        &kd,
                        &nRhs,
                        reinterpret_cast<const double *>(Ab),
                        &ldAb,
                        reinterpret_cast<double *>(B),
                        &ldB,
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

#endif // CXXLAPACK_INTERFACE_PBTRS_TCC
