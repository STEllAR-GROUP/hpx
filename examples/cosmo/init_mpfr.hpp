//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_INIT_MPFR_OCT_29_2010_1134AM)
#define HPX_COMPONENTS_AMR_INIT_MPFR_OCT_29_2010_1134AM

#if MPFR_FOUND != 0
#include "mpreal.h"
#endif

namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    // initialize mpreal default precision
    struct init_mpfr
    {
        init_mpfr(bool init_allocators = false)
        {
#if MPFR_FOUND != 0
            mpfr::mpreal::set_default_prec(64);
#if MPFR_USE_NED_ALLOCATOR != 0
            if (init_allocators) {
                mp_set_memory_functions(mpfr::mpreal::mpreal_allocate, 
                    mpfr::mpreal::mpreal_reallocate, mpfr::mpreal::mpreal_free);
            }
#endif
#endif
        }
    };

}}}

#endif
