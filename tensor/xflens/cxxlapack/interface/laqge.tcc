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

#ifndef CXXLAPACK_INTERFACE_LAQGE_TCC
#define CXXLAPACK_INTERFACE_LAQGE_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
laqge(IndexType     m,
      IndexType     n,
      float         *A,
      IndexType     ldA,
      const float   *r,
      const float   *c,
      const float   &rowCond,
      const float   &colCond,
      const float   &maxA,
      char          &equed)
{
    CXXLAPACK_DEBUG_OUT("slaqge");

    LAPACK_IMPL(slaqge)(&m,
                        &n,
                        A,
                        &ldA,
                        r,
                        c,
                        &rowCond,
                        &colCond,
                        &maxA,
                        &equed);

}


template <typename IndexType>
void
laqge(IndexType     m,
      IndexType     n,
      double        *A,
      IndexType     ldA,
      const double  *r,
      const double  *c,
      const double  &rowCond,
      const double  &colCond,
      const double  &maxA,
      char          &equed)
{
    CXXLAPACK_DEBUG_OUT("dlaqge");

    LAPACK_IMPL(dlaqge)(&m,
                        &n,
                        A,
                        &ldA,
                        r,
                        c,
                        &rowCond,
                        &colCond,
                        &maxA,
                        &equed);

}

template <typename IndexType>
void
laqge(const IndexType       m,
      const IndexType       n,
      std::complex<float >  *A,
      const IndexType       ldA,
      const float           *r,
      const float           *c,
      const float           &rowCond,
      const float           &colCond,
      const float           &maxA,
      char                  &equed)
{
    CXXLAPACK_DEBUG_OUT("claqge");

    LAPACK_IMPL(claqge)(&m,
                        &n,
                        reinterpret_cast<float  *>(A),
                        &ldA,
                        r,
                        c,
                        &rowCond,
                        &colCond,
                        &maxA,
                        &equed);
}

template <typename IndexType>
void
laqge(const IndexType       m,
      const IndexType       n,
      std::complex<double>  *A,
      const IndexType       ldA,
      const double          *r,
      const double          *c,
      const double          &rowCond,
      const double          &colCond,
      const double          &maxA,
      char                  &equed)
{
    CXXLAPACK_DEBUG_OUT("zlaqge");

    LAPACK_IMPL(zlaqge)(&m,
                        &n,
                        reinterpret_cast<double *>(A),
                        &ldA,
                        r,
                        c,
                        &rowCond,
                        &colCond,
                        &maxA,
                        &equed);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAQGE_TCC
