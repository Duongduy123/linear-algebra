/*
 *   Copyright (c) 2013, Klaus Pototzky
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

#ifndef CXXBLAS_LEVEL1_AXPBY_H
#define CXXBLAS_LEVEL1_AXPBY_H 1

#include "../typedefs.h"

#define HAVE_CXXBLAS_AXPBY 1

namespace cxxblas {

template <typename IndexType, typename ALPHA, typename X,
          typename BETA, typename Y>
    void
    axpby(IndexType n, const ALPHA &alpha, const X *x, IndexType incX,
                       const BETA &beta, Y *y, IndexType incY);

#ifdef HAVE_CBLAS_AXPBY
// saxpy
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    axpby(IndexType n, const float &alpha,
          const float *x, IndexType incX,
          const float &beta,
          float *y, IndexType incY);

// daxpy
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    axpby(IndexType n, const double &alpha,
          const double *x, IndexType incX,
          const double &beta,
          double *y, IndexType incY);

// caxpy
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    axpby(IndexType n, const ComplexFloat &alpha,
          const ComplexFloat *x, IndexType incX,
          const ComplexFloat &beta,
          ComplexFloat *y, IndexType incY);

// zaxpy
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    axpby(IndexType n, const ComplexDouble &alpha,
          const ComplexDouble *x, IndexType incX,
          const ComplexDouble &beta,
          ComplexDouble *y, IndexType incY);

#endif // HAVE_CBLAS_AXPBY

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1_AXPBY_H
