//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_STATIC_REINIT_OCT_25_2012_0921AM)
#define HPX_UTIL_STATIC_REINIT_OCT_25_2012_0921AM

#include <hpx/config.hpp>
#include <hpx/util/function.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // This is a global API allowing to register functions to be called before
    // the runtime system is about to start and after the runtime system has
    // been terminated. This is used to initialize/reinitialize all
    // singleton instances.
    HPX_API_EXPORT void reinit_register(
        util::function_nonser<void()> const& construct,
        util::function_nonser<void()> const& destruct);

    // Invoke all globally registered construction functions
    HPX_API_EXPORT void reinit_construct();

    // Invoke all globally registered destruction functions
    HPX_API_EXPORT void reinit_destruct();

    // just a little helper to invoke the constructors and destructors
    struct reinit_helper
    {
        reinit_helper()
        {
        }
        ~reinit_helper ()
        {
            reinit_destruct();
        }
    };
}}

#endif


