//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MAIN_SEP_20_2014_1130AM)
#define HPX_MAIN_SEP_20_2014_1130AM

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>

#if defined(HPX_HAVE_STATIC_LINKING)
#include <hpx/hpx_main_impl.hpp>
#endif

// We support redefining the plain C-main provided by the user to be executed
// as the first HPX-thread (equivalent to hpx_main()). This is implemented by
// a macro redefining main, so we disable it by default.
#define main hpx_startup::user_main

#endif
