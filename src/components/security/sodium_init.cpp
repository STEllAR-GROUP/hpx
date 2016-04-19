//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SODIUM)
#include <sodium.h>

namespace hpx { namespace components { namespace security
{
    // sodium_init has to be called once in a thread safe environment
    struct init_sodium
    {
        init_sodium()
        {
            sodium_init();
        }
    };

    init_sodium init;
}}}

#endif
