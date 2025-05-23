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

#ifndef CXXLAPACK_INTERFACE_LA_GBRCOND_TCC
#define CXXLAPACK_INTERFACE_LA_GBRCOND_TCC

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
double
la_gbrCond(char                  trans,
           IndexType             n,
           IndexType             kl,
           IndexType             ku,
           const float           *Ab,
           IndexType             ldAb,
           const float           *Afb,
           IndexType             ldAfb,
           const IndexType       *iPiv,
           IndexType             cmode,
           const float           *c,
           IndexType             &info,
           float                 *work,
           IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("sla_gbrcond");

    return LAPACK_IMPL(sla_gbrcond)(&trans,
                                    &n,
                                    &kl,
                                    &ku,
                                    Ab,
                                    &ldAb,
                                    Afb,
                                    &ldAfb,
                                    iPiv,
                                    &cmode,
                                    c,
                                    &info,
                                    work,
                                    iWork);
}

template <typename IndexType>
double
la_gbrCond(char                  trans,
           IndexType             n,
           IndexType             kl,
           IndexType             ku,
           const double          *Ab,
           IndexType             ldAb,
           const double          *Afb,
           IndexType             ldAfb,
           const IndexType       *iPiv,
           IndexType             cmode,
           const double          *c,
           IndexType             &info,
           double                *work,
           IndexType             *iWork)
{
    CXXLAPACK_DEBUG_OUT("dla_gbrcond");

    return LAPACK_IMPL(dla_gbrcond)(&trans,
                                    &n,
                                    &kl,
                                    &ku,
                                    Ab,
                                    &ldAb,
                                    Afb,
                                    &ldAfb,
                                    iPiv,
                                    &cmode,
                                    c,
                                    &info,
                                    work,
                                    iWork);
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LA_GBRCOND_TCC
