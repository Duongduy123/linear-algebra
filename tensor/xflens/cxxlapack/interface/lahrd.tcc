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

#ifndef CXXLAPACK_INTERFACE_LAHRD_TCC
#define CXXLAPACK_INTERFACE_LAHRD_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
lahrd(IndexType             n,
      IndexType             k,
      IndexType             nb,
      float                 *A,
      IndexType             ldA,
      float                 *tau,
      float                 *T,
      IndexType             ldT,
      float                 *Y,
      IndexType             ldY)
{
    CXXLAPACK_DEBUG_OUT("slahrd");

    LAPACK_IMPL(slahrd)(&n,
                        &k,
                        &nb,
                        A,
                        &ldA,
                        tau,
                        T,
                        &ldT,
                        Y,
                        &ldY);
}


template <typename IndexType>
void
lahrd(IndexType             n,
      IndexType             k,
      IndexType             nb,
      double                *A,
      IndexType             ldA,
      double                *tau,
      double                *T,
      IndexType             ldT,
      double                *Y,
      IndexType             ldY)
{
    CXXLAPACK_DEBUG_OUT("dlahrd");

    LAPACK_IMPL(dlahrd)(&n,
                        &k,
                        &nb,
                        A,
                        &ldA,
                        tau,
                        T,
                        &ldT,
                        Y,
                        &ldY);
}


template <typename IndexType>
void
lahrd(IndexType             n,
      IndexType             k,
      IndexType             nb,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *tau,
      std::complex<float >  *T,
      IndexType             ldT,
      std::complex<float >  *Y,
      IndexType             ldY)
{
    CXXLAPACK_DEBUG_OUT("clahrd");

    LAPACK_IMPL(clahrd)(&n,
                        &k,
                        &nb,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        reinterpret_cast<float  *>(tau),
                        reinterpret_cast<float  *>(T),
                        &ldT,
                        reinterpret_cast<float  *>(Y),
                        &ldY);
}

template <typename IndexType>
void
lahrd(IndexType             n,
      IndexType             k,
      IndexType             nb,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *tau,
      std::complex<double>  *T,
      IndexType             ldT,
      std::complex<double>  *Y,
      IndexType             ldY)
{
    CXXLAPACK_DEBUG_OUT("zlahrd");

    LAPACK_IMPL(zlahrd)(&n,
                        &k,
                        &nb,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        reinterpret_cast<double *>(tau),
                        reinterpret_cast<double *>(T),
                        &ldT,
                        reinterpret_cast<double *>(Y),
                        &ldY);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAHRD_TCC
