/*
 *   Copyright (c) 2009, Michael Lehn
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

#ifndef CXXBLAS_LEVEL1EXTENSIONS_IMAX1_TCC
#define CXXBLAS_LEVEL1EXTENSIONS_IMAX1_TCC 1

#include <complex>
#include <cmath>
#include "../cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename X>
void
imax1_generic(IndexType n, const std::complex<X> *x, IndexType incX,
              IndexType &iAbsMaxX)
{
    CXXBLAS_DEBUG_OUT("imax1_generic");

    using std::abs;

    iAbsMaxX = 0;
    X absMaxX = abs(x[iAbsMaxX]);
    for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
        if (abs(x[iX])>absMaxX) {
            iAbsMaxX = i;
            absMaxX = abs(x[iX]);
        }
    }
}


template <typename IndexType, typename X>
void
imax1(IndexType n, const X *x, IndexType incX, IndexType &iAbsMaxX)
{
    if (incX<0) {
        x -= incX*(n-1);
    }
    if (n<=0) {
        iAbsMaxX = -1;
        return;
    }
    imax1_generic(n, x, incX, iAbsMaxX);
}

template <typename IndexType, typename X>
IndexType
imax1(IndexType n, const X *x, IndexType incX)
{
    IndexType iAbsMaxX = IndexType(0);

    imax1(n, x, incX, iAbsMaxX);
    return iAbsMaxX;
}


} // namespace cxxblas

#endif // CXXBLAS_LEVEL1EXTENSIONS_IMAX1_TCC
