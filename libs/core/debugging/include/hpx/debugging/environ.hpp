//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

// The 'environ' should be declared in some cases. E.g. Linux man page says:
// (This variable must be declared in the user program, but is declared in
// the header file unistd.h in case the header files came from libc4 or libc5,
// and in case they came from glibc and _GNU_SOURCE was defined.)
// To be safe, declare it here.

#if defined(__linux) || defined(linux) || defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#elif defined(__APPLE__)
// It appears that on Mac OS X the 'environ' variable is not
// available to dynamically linked libraries.
// See: http://article.gmane.org/gmane.comp.lib.boost.devel/103843
// See: http://lists.gnu.org/archive/html/bug-guile/2004-01/msg00013.html
#include <unistd.h>
// The proper include for this is crt_externs.h, however it's not
// available on iOS. The right replacement is not known. See
// https://svn.boost.org/trac/boost/ticket/5053
extern "C" {
extern char*** _NSGetEnviron(void);
}
#define environ (*_NSGetEnviron())
#elif defined(HPX_WINDOWS)
#include <winsock2.h>
#define environ _environ
#elif defined(__FreeBSD__)
// On FreeBSD the environment is available for executables only, so needs to be
// handled explicitly (e.g. see hpx_init_impl.hpp)
// The variable is defined in .../runtime_local/src/custom_exception_info.cpp
extern HPX_CORE_EXPORT char** freebsd_environ;
#else
extern char** environ;
#endif
