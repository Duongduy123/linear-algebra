/*
 *   Copyright (c) 2010, Michael Lehn
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

#ifndef CXXBLAS_LEVEL1EXTENSIONS_GECOPY_TCC
#define CXXBLAS_LEVEL1EXTENSIONS_GECOPY_TCC 1

#include <algorithm>
#include <cassert>
#include "../cxxblas.h"

namespace cxxblas {

//
//  B = A, B = A^T or B = A^H
//
template <typename IndexType, typename MA, typename MB>
void
gecopy(StorageOrder order,
       Transpose trans, IndexType m, IndexType n,
       const MA *A, IndexType ldA,
       MB *B, IndexType ldB)
{
    CXXBLAS_DEBUG_OUT("gecopy_generic");

    if (order==ColMajor) {
        gecopy(RowMajor, trans, n, m, A, ldA, B, ldB);
        return;
    }
    if (trans==NoTrans) {
        if ((ldA==n) && (ldB==n)) {
            copy(m*n, A, IndexType(1), B, IndexType(1));
            return;
        } else {
            for (IndexType i=0; i<m; ++i) {
                copy(n, A+i*ldA, IndexType(1), B+i*ldB, IndexType(1));
            }
            return;
        }
    } else if (trans==Conj) {
        if ((ldA==n) && (ldB==n)) {
            ccopy(m*n, A, IndexType(1), B, IndexType(1));
            return;
        } else {
            for (IndexType i=0; i<m; ++i) {
                ccopy(n, A+i*ldA, IndexType(1), B+i*ldB, IndexType(1));
            }
            return;
        }
    } else if (trans==Trans) {
        for (IndexType i=0; i<m; ++i) {
            copy(n, A+i, ldA, B+i*ldB, IndexType(1));
        }
        return;
    } else if (trans==ConjTrans) {
        for (IndexType i=0; i<m; ++i) {
            ccopy(n, A+i, ldA, B+i*ldB, IndexType(1));
        }
        return;
    } else {
        ASSERT(0);
    }
}

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1EXTENSIONS_GECOPY_TCC
