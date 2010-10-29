//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_INIT_MPFR_OCT_29_2010_1134AM)
#define HPX_COMPONENTS_AMR_INIT_MPFR_OCT_29_2010_1134AM

#if defined(MPFR_FOUND)
#include "mpreal.h"
#endif

namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    // initialize mpreal default precision
    struct init_mpfr
    {
        init_mpfr()
        {
#if defined(MPFR_FOUND)
            mpfr::mpreal::set_default_prec(128);
#endif
        }
    };

}}}

#endif
