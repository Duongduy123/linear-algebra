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

#ifndef CXXLAPACK_INTERFACE_LAQHB_TCC
#define CXXLAPACK_INTERFACE_LAQHB_TCC 1

#include <iostream>
#include "interface.h"
#include "../netlib/netlib.h"

namespace cxxlapack {

template <typename IndexType>
void
laqhb(char                  uplo,
      IndexType             n,
      IndexType             kd,
      std::complex<float >  *Ab,
      IndexType             ldAb,
      const float           *s,
      float                 scond,
      float                 amax,
      char                  &equed)
{
    CXXLAPACK_DEBUG_OUT("claqhb");

    LAPACK_IMPL(claqhb)(&uplo,
                        &n,
                        &kd,
                        reinterpret_cast<float  *>(Ab),
                        &ldAb,
                        s,
                        &scond,
                        &amax,
                        &equed);

}

template <typename IndexType>
void
laqhb(char                  uplo,
      IndexType             n,
      IndexType             kd,
      std::complex<double>  *Ab,
      IndexType             ldAb,
      const double          *s,
      double                scond,
      double                amax,
      char                  &equed)
{
    CXXLAPACK_DEBUG_OUT("zlaqhb");

    LAPACK_IMPL(zlaqhb)(&uplo,
                        &n,
                        &kd,
                        reinterpret_cast<double *>(Ab),
                        &ldAb,
                        s,
                        &scond,
                        &amax,
                        &equed);

}

} // namespace cxxlapack

#endif // CXXLAPACK_INTERFACE_LAQHB_TCC
