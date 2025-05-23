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

#ifndef CXXLAPACK_INTERFACE_LARRE_TCC
#define CXXLAPACK_INTERFACE_LARRE_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
IndexType
larre(char                  range,
      IndexType             n,
      float                 &vl,
      float                 &vu,
      IndexType             il,
      IndexType             iu,
      float                 *d,
      float                 *e,
      float                 *e2,
      float                 rtol1,
      float                 rtol2,
      float                 spltol,
      IndexType             &nsplit,
      IndexType             *isplit,
      IndexType             &m,
      float                 *w,
      float                 *werr,
      float                 *wgap,
      IndexType             *iblock,
      IndexType             *indexw,
      float                 *gers,
      float                 &pivmin,
      float                 *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("slarre");

    IndexType info;
    LAPACK_IMPL(slarre)(&range,
                        &n,
                        &vl,
                        &vu,
                        &il,
                        &iu,
                        d,
                        e,
                        e2,
                        &rtol1,
                        &rtol2,
                        &spltol,
                        &nsplit,
                        isplit,
                        &m,
                        w,
                        werr,
                        wgap,
                        iblock,
                        indexw,
                        gers,
                        &pivmin,
                        work,
                        iWork,
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
larre(char                  range,
      IndexType             n,
      double                &vl,
      double                &vu,
      IndexType             il,
      IndexType             iu,
      double                *d,
      double                *e,
      double                *e2,
      double                rtol1,
      double                rtol2,
      double                spltol,
      IndexType             &nsplit,
      IndexType             *isplit,
      IndexType             &m,
      double                *w,
      double                *werr,
      double                *wgap,
      IndexType             *iblock,
      IndexType             *indexw,
      double                *gers,
      double                &pivmin,
      double                *work,
      IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("dlarre");

    IndexType info;
    LAPACK_IMPL(dlarre)(&range,
                        &n,
                        &vl,
                        &vu,
                        &il,
                        &iu,
                        d,
                        e,
                        e2,
                        &rtol1,
                        &rtol2,
                        &spltol,
                        &nsplit,
                        isplit,
                        &m,
                        w,
                        werr,
                        wgap,
                        iblock,
                        indexw,
                        gers,
                        &pivmin,
                        work,
                        iWork,
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

#endif // CXXLAPACK_INTERFACE_LARRE_TCC
