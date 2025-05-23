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

#ifndef CXXBLAS_LEVEL2_HPR_TCC
#define CXXBLAS_LEVEL2_HPR_TCC 1

#include <complex>
#include "../cxxblas.h"

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename VX, typename MA>
void
hpr_generic(StorageOrder order, StorageUpLo upLo, Transpose conjugateA,
            IndexType n,
            const ALPHA &alpha,
            const VX *x, IndexType incX,
            MA *A)
{
    if (alpha==ALPHA(0)) {
        return;
    }
    if (order==ColMajor) {
        upLo = (upLo==Upper) ? Lower : Upper;
        conjugateA = Transpose(conjugateA^Conj);
    }
    #ifdef CXXBLAS_USE_XERBLA
        // insert error check here
    #endif
    if (upLo==Upper) {
        if (conjugateA==Conj) {
            for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                axpy_generic(n-i, alpha*conjugate(x[iX]),
                                  x+iX, incX,
                                  A+i*(2*n-i+1)/2, IndexType(1));
            }
        } else {
            for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                acxpy_generic(n-i, alpha*x[iX],
                                   x+iX, incX,
                                   A+i*(2*n-i+1)/2, IndexType(1));
            }
        }
        for (IndexType i=0; i<n; ++i) {
            A[i+i*(2*n-i-1)/2] = cxxblas::real(A[i+i*(2*n-i-1)/2]);
        }
    } else {
        if (conjugateA==Conj) {
            for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                axpy_generic(i+1, alpha*conjugate(x[iX]),
                                  x, incX,
                                  A+i*(i+1)/2, IndexType(1));
            }
        } else {
            for (IndexType i=0, iX=0; i<n; ++i, iX+=incX) {
                acxpy_generic(i+1, alpha*x[iX],
                                   x, incX,
                                   A+i*(i+1)/2, IndexType(1));
            }
        }
        for (IndexType i=0; i<n; ++i) {
            A[i+i*(i+1)/2] = cxxblas::real(A[i+i*(i+1)/2]);
        }
    }
}

template <typename IndexType, typename ALPHA, typename VX, typename MA>
void
hpr(StorageOrder order, StorageUpLo upLo,
    IndexType n,
    const ALPHA &alpha,
    const VX *x, IndexType incX,
    MA *A)
{
    CXXBLAS_DEBUG_OUT("hpr_generic");

    if (incX<0) {
        x -= incX*(n-1);
    }
    hpr_generic(order, upLo, NoTrans, n, alpha, x, incX, A);
}


#ifdef HAVE_CBLAS

// chpr
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
hpr(StorageOrder order,   StorageUpLo upLo,
      IndexType n,
      float alpha,
      const ComplexFloat *x, IndexType incX,
      ComplexFloat *A)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_chpr");

    cblas_chpr(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
               n,
               alpha,
               reinterpret_cast<const float *>(x), incX,
               reinterpret_cast<float *>(A));
}

// zhpr
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
hpr(StorageOrder order,   StorageUpLo upLo,
      IndexType n,
      double alpha,
      const ComplexDouble *x, IndexType incX,
      ComplexDouble *A)
{
    CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zhpr");

    cblas_zhpr(CBLAS::getCblasType(order), CBLAS::getCblasType(upLo),
               n,
               alpha,
               reinterpret_cast<const double *>(x), incX,
               reinterpret_cast<double *>(A));
}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_HPR_TCC
