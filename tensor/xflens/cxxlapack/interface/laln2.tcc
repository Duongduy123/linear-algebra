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

#ifndef CXXLAPACK_INTERFACE_LALN2_TCC
#define CXXLAPACK_INTERFACE_LALN2_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
laln2(bool              trans,
      IndexType         na,
      IndexType         nw,
      const float       &sMin,
      const float       &ca,
      const float       *A,
      IndexType         ldA,
      const float       &d1,
      const float       &d2,
      const float       *B,
      IndexType         ldB,
      const float       &wr,
      const float       &wi,
      float             *X,
      IndexType         ldX,
      float             &scale,
      float             &normX)
{
    CXXLAPACK_DEBUG_OUT("slaln2");

    IndexType info;
    IndexType trans_ = trans;
    LAPACK_IMPL(slaln2)(&trans_,
                        &na,
                        &nw,
                        &sMin,
                        &ca,
                        A,
                        &ldA,
                        &d1,
                        &d2,
                        B,
                        &ldB,
                        &wr,
                        &wi,
                        X,
                        &ldX,
                        &scale,
                        &normX,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
laln2(bool              trans,
      IndexType         na,
      IndexType         nw,
      const double      &sMin,
      const double      &ca,
      const double      *A,
      IndexType         ldA,
      const double      &d1,
      const double      &d2,
      const double      *B,
      IndexType         ldB,
      const double      &wr,
      const double      &wi,
      double            *X,
      IndexType         ldX,
      double            &scale,
      double            &normX)
{
    CXXLAPACK_DEBUG_OUT("dlaln2");

    IndexType info;
    IndexType trans_ = trans;
    LAPACK_IMPL(dlaln2)(&trans_,
                        &na,
                        &nw,
                        &sMin,
                        &ca,
                        A,
                        &ldA,
                        &d1,
                        &d2,
                        B,
                        &ldB,
                        &wr,
                        &wi,
                        X,
                        &ldX,
                        &scale,
                        &normX,
                        &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LALN2_TCC
