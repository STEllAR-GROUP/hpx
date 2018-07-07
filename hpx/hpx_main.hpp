//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2018 Nikunj Gupta
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_HPX_MAIN_HPP
#define HPX_HPX_MAIN_HPP

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>

// We support different implementation depending upon the Operating
// System in use.
#if defined(__linux) || defined(__linux__) || defined(linux)

namespace hpx_start {
    // include_libhpx_wrap here is an override for the one present in
    // src/hpx_wrap.cpp. The value of this variable defines if we need
    // to change the program's entry point or not.
    extern bool include_libhpx_wrap;
    bool include_libhpx_wrap = true;
}

#else

#if defined(HPX_HAVE_STATIC_LINKING)
#include <hpx/hpx_main_impl.hpp>
#endif

// We support redefining the plain C-main provided by the user to be executed
// as the first HPX-thread (equivalent to hpx_main()). This is implemented by
// a macro redefining main, so we disable it by default.
#define main hpx_startup::user_main
#endif

#endif /*HPX_HPX_MAIN_HPP*/
