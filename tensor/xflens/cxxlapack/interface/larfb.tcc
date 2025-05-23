/*
 *   Copyright (c) 2012, Michael Lehn
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

#ifndef CXXLAPACK_INTERFACE_LARFB_TCC
#define CXXLAPACK_INTERFACE_LARFB_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
larfb(char              side,
      char              trans,
      char              direct,
      char              storev,
      IndexType         m,
      IndexType         n,
      IndexType         k,
      const float       *V,
      IndexType         ldV,
      const float       *T,
      IndexType         ldT,
      float             *C,
      IndexType         ldC,
      float             *work,
      const IndexType   ldWork)
{
    CXXLAPACK_DEBUG_OUT("slarfb");

    LAPACK_IMPL(slarfb)(&side,
                        &trans,
                        &direct,
                        &storev,
                        &m,
                        &n,
                        &k,
                        V,
                        &ldV,
                        T,
                        &ldT,
                        C,
                        &ldC,
                        work,
                        &ldWork);
}


template <typename IndexType>
void
larfb(char              side,
      char              trans,
      char              direct,
      char              storev,
      IndexType         m,
      IndexType         n,
      IndexType         k,
      const double      *V,
      IndexType         ldV,
      const double      *T,
      IndexType         ldT,
      double            *C,
      IndexType         ldC,
      double            *work,
      const IndexType   ldWork)
{
    CXXLAPACK_DEBUG_OUT("dlarfb");

    LAPACK_IMPL(dlarfb)(&side,
                        &trans,
                        &direct,
                        &storev,
                        &m,
                        &n,
                        &k,
                        V,
                        &ldV,
                        T,
                        &ldT,
                        C,
                        &ldC,
                        work,
                        &ldWork);
}

template <typename IndexType>
void
larfb(char                          side,
      char                          trans,
      char                          direct,
      char                          storev,
      IndexType                     m,
      IndexType                     n,
      IndexType                     k,
      const std::complex<float >    *V,
      IndexType                     ldV,
      const std::complex<float >    *T,
      IndexType                     ldT,
      std::complex<float >          *C,
      IndexType                     ldC,
      std::complex<float >          *work,
      IndexType                     ldWork)
{
    CXXLAPACK_DEBUG_OUT("clarfb");

    LAPACK_IMPL(clarfb)(&side,
                        &trans,
                        &direct,
                        &storev,
                        &m,
                        &n,
                        &k,
                        reinterpret_cast<const float  *>(V),
                        &ldV,
                        reinterpret_cast<const float  *>(T),
                        &ldT,
                        reinterpret_cast<float  *>(C),
                        &ldC,
                        reinterpret_cast<float  *>(work),
                        &ldWork);
}

template <typename IndexType>
void
larfb(char                          side,
      char                          trans,
      char                          direct,
      char                          storev,
      IndexType                     m,
      IndexType                     n,
      IndexType                     k,
      const std::complex<double>    *V,
      IndexType                     ldV,
      const std::complex<double>    *T,
      IndexType                     ldT,
      std::complex<double>          *C,
      IndexType                     ldC,
      std::complex<double>          *work,
      IndexType                     ldWork)
{
    CXXLAPACK_DEBUG_OUT("zlarfb");

    LAPACK_IMPL(zlarfb)(&side,
                        &trans,
                        &direct,
                        &storev,
                        &m,
                        &n,
                        &k,
                        reinterpret_cast<const double *>(V),
                        &ldV,
                        reinterpret_cast<const double *>(T),
                        &ldT,
                        reinterpret_cast<double *>(C),
                        &ldC,
                        reinterpret_cast<double *>(work),
                        &ldWork);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LARFB_TCC
