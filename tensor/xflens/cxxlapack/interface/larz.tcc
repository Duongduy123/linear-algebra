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

#ifndef CXXLAPACK_INTERFACE_LARZ_TCC
#define CXXLAPACK_INTERFACE_LARZ_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
larz(char           side,
     IndexType      m,
     IndexType      n,
     IndexType      l,
     const float    *V,
     IndexType      incV,
     float          tau,
     float          *C,
     IndexType      ldC,
     float          *work)
{
    CXXLAPACK_DEBUG_OUT("slarz");

    LAPACK_IMPL(slarz)(&side,
                       &m,
                       &n,
                       &l,
                       V,
                       &incV,
                       &tau,
                       C,
                       &ldC,
                       work);
}


template <typename IndexType>
void
larz(char           side,
     IndexType      m,
     IndexType      n,
     IndexType      l,
     const double   *V,
     IndexType      incV,
     double         tau,
     double         *C,
     IndexType      ldC,
     double         *work)
{
    CXXLAPACK_DEBUG_OUT("dlarz");

    LAPACK_IMPL(dlarz)(&side,
                       &m,
                       &n,
                       &l,
                       V,
                       &incV,
                       &tau,
                       C,
                       &ldC,
                       work);
}

template <typename IndexType>
void
larz(char                        side,
     IndexType                   m,
     IndexType                   n,
     IndexType                   l,
     const std::complex<float >  *V,
     IndexType                   incV,
     const std::complex<float >  &tau,
     std::complex<float >        *C,
     IndexType                   ldC,
     std::complex<float >        *work)
{
    CXXLAPACK_DEBUG_OUT("clarz");

    LAPACK_IMPL(clarz)(&side,
                       &m,
                       &n,
                       &l,
                       reinterpret_cast<const float  *>(V),
                       &incV,
                       reinterpret_cast<const float  *>(&tau),
                       reinterpret_cast<float  *>(C),
                       &ldC,
                       reinterpret_cast<float  *>(work));
}

template <typename IndexType>
void
larz(char                        side,
     IndexType                   m,
     IndexType                   n,
     IndexType                   l,
     const std::complex<double>  *V,
     IndexType                   incV,
     const std::complex<double>  &tau,
     std::complex<double>        *C,
     IndexType                   ldC,
     std::complex<double>        *work)
{
    CXXLAPACK_DEBUG_OUT("zlarz");

    LAPACK_IMPL(zlarz)(&side,
                       &m,
                       &n,
                       &l,
                       reinterpret_cast<const double *>(V),
                       &incV,
                       reinterpret_cast<const double *>(&tau),
                       reinterpret_cast<double *>(C),
                       &ldC,
                       reinterpret_cast<double *>(work));
}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LARZ_TCC 1
