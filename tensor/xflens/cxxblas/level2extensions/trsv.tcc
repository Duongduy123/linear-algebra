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

#ifndef CXXBLAS_LEVEL2EXTENSIONS_TRSV_TCC
#define CXXBLAS_LEVEL2EXTENSIONS_TRSV_TCC 1

#include <complex>
#include "../cxxblas.h"

namespace cxxblas {

#ifdef HAVE_CBLAS

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const float *A, IndexType ldA,
     ComplexFloat *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("trsv (extension)");

    float *x_ = reinterpret_cast<float *>(x);

    trsv(order, upLo, transA, diag, n, A, ldA, x_,   2*incX);
    trsv(order, upLo, transA, diag, n, A, ldA, x_+1, 2*incX);
}

template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const double *A, IndexType ldA,
     ComplexDouble *x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("trsv (extension)");

    double *x_ = reinterpret_cast<double *>(x);

    trsv(order, upLo, transA, diag, n, A, ldA, x_,   2*incX);
    trsv(order, upLo, transA, diag, n, A, ldA, x_+1, 2*incX);
}

#endif

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2EXTENSIONS_TRSV_TCC
