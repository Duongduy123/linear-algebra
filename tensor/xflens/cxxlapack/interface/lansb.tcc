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

#ifndef CXXLAPACK_INTERFACE_LANSB_TCC
#define CXXLAPACK_INTERFACE_LANSB_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
float
lansb(char                  norm,
      char                  uplo,
      IndexType             n,
      IndexType             k,
      const float           *Ab,
      IndexType             ldAb,
      float                 *work)
{
    CXXLAPACK_DEBUG_OUT("slansb");

    return LAPACK_IMPL(slansb)(&norm,
                               &uplo,
                               &n,
                               &k,
                               Ab,
                               &ldAb,
                               work);
}


template <typename IndexType>
double
lansb(char                  norm,
      char                  uplo,
      IndexType             n,
      IndexType             k,
      const double          *Ab,
      IndexType             ldAb,
      double                *work)
{
    CXXLAPACK_DEBUG_OUT("dlansb");

    return LAPACK_IMPL(dlansb)(&norm,
                               &uplo,
                               &n,
                               &k,
                               Ab,
                               &ldAb,
                               work);
}

template  <typename IndexType>
float
lansb(char                        norm,
      char                        uplo,
      IndexType                   n,
      IndexType                   k,
      const std::complex<float >  *Ab,
      IndexType                   ldAb,
      float                       *work)
{
    CXXLAPACK_DEBUG_OUT("clansb");

    return LAPACK_IMPL(clansb)(&norm,
                               &uplo,
                               &n,
                               &k,
                               reinterpret_cast<const float  *>(Ab),
                               &ldAb,
                               work);
}

template <typename IndexType>
double
lansb(char                        norm,
      char                        uplo,
      IndexType                   n,
      IndexType                   k,
      const std::complex<double>  *Ab,
      IndexType                   ldAb,
      double                      *work)
{
    CXXLAPACK_DEBUG_OUT("zlansb");

    return LAPACK_IMPL(zlansb)(&norm,
                               &uplo,
                               &n,
                               &k,
                               reinterpret_cast<const double *>(Ab),
                               &ldAb,
                               work);
}


} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LANSB_TCC
