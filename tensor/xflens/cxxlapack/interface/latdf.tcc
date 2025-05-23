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

#ifndef CXXLAPACK_INTERFACE_LATDF_TCC
#define CXXLAPACK_INTERFACE_LATDF_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
latdf(IndexType             ijob,
      IndexType             n,
      const float           *Z,
      IndexType             ldZ,
      float                 *rhs,
      float                 &rdsum,
      float                 &rdscal,
      const IndexType       *iPiv,
      const IndexType       *jpiv)
{
    CXXLAPACK_DEBUG_OUT("slatdf");

    LAPACK_IMPL(slatdf)(&ijob,
                        &n,
                        Z,
                        &ldZ,
                        rhs,
                        &rdsum,
                        &rdscal,
                        iPiv,
                        jpiv);
}


template <typename IndexType>
void
latdf(IndexType             ijob,
      IndexType             n,
      const double          *Z,
      IndexType             ldZ,
      double                *rhs,
      double                &rdsum,
      double                &rdscal,
      const IndexType       *iPiv,
      const IndexType       *jpiv)
{
    CXXLAPACK_DEBUG_OUT("dlatdf");

    LAPACK_IMPL(dlatdf)(&ijob,
                        &n,
                        Z,
                        &ldZ,
                        rhs,
                        &rdsum,
                        &rdscal,
                        iPiv,
                        jpiv);
}

template <typename IndexType>
void
latdf(IndexType                  ijob,
      IndexType                  n,
      const std::complex<float > *Z,
      IndexType                  ldZ,
      std::complex<float >       *rhs,
      float                      &rdsum,
      float                      &rdscal,
      const IndexType           *iPiv,
      const IndexType           *jpiv)
{
    CXXLAPACK_DEBUG_OUT("clatdf");

    LAPACK_IMPL(clatdf)(&ijob,
                        &n,
                        reinterpret_cast<const float  *>(Z),
                        &ldZ,
                        reinterpret_cast<float  *>(rhs),
                        &rdsum,
                        &rdscal,
                        iPiv,
                        jpiv);
}

template <typename IndexType>
void
latdf(IndexType                  ijob,
      IndexType                  n,
      const std::complex<double> *Z,
      IndexType                  ldZ,
      std::complex<double>       *rhs,
      double                     &rdsum,
      double                     &rdscal,
      const IndexType           *iPiv,
      const IndexType           *jpiv)
{
    CXXLAPACK_DEBUG_OUT("zlatdf");

    LAPACK_IMPL(zlatdf)(&ijob,
                        &n,
                        reinterpret_cast<const double *>(Z),
                        &ldZ,
                        reinterpret_cast<double *>(rhs),
                        &rdsum,
                        &rdscal,
                        iPiv,
                        jpiv);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LATDF_TCC
