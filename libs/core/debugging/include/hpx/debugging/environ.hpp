//  Copyright (c) 2021-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/debugging/macros.hpp>

// The 'environ' should be declared in some cases. E.g. Linux man page says:
// (This variable must be declared in the user program, but is declared in
// the header file unistd.h in case the header files came from libc4 or libc5,
// and in case they came from glibc and _GNU_SOURCE was defined.)
// To be safe, declare it here.

#if defined(__linux) || defined(linux) || defined(__linux__)
// this case is handled in debugging/macros.hpp
#elif defined(__APPLE__)
// this case is handled in debugging/macros.hpp
#elif defined(HPX_WINDOWS)
// this case is handled in debugging/macros.hpp
#elif defined(__FreeBSD__)
// On FreeBSD the environment is available for executables only, so needs to be
// handled explicitly (e.g. see hpx_init_impl.hpp)
// The variable is defined in debugging/src/print.cpp
HPX_CORE_MODULE_EXPORT char** freebsd_environ;
#else
// this case is handled in debugging/macros.hpp
extern char** environ;
#endif
