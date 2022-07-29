////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

// clang-format off
#if defined(HPX_MSVC)
  #include <hpx/hardware/timestamp/msvc.hpp>
#elif defined(__amd64__) || defined(__amd64) || defined(__x86_64__) ||         \
    defined(__x86_64) || defined(_M_X64)
    #if defined(HPX_HAVE_RDTSC) || defined(HPX_HAVE_RDTSCP)
        #include <hpx/hardware/timestamp/linux_x86_64.hpp>
    #else
        #include <hpx/hardware/timestamp/linux_generic.hpp>
    #endif
#elif defined(i386) || defined(__i386__) || defined(__i486__) ||               \
    defined(__i586__) || defined(__i686__) || defined(__i386) ||               \
    defined(_M_IX86) || defined(__X86__) || defined(_X86_) ||                  \
    defined(__THW_INTEL__) || defined(__I86__) || defined(__INTEL__)
    #if defined(HPX_HAVE_RDTSC) || defined(HPX_HAVE_RDTSCP)
        #include <hpx/hardware/timestamp/linux_x86_32.hpp>
    #else
        #include <hpx/hardware/timestamp/linux_generic.hpp>
    #endif
#elif (defined(__ANDROID__) && defined(ANDROID))
    #include <hpx/hardware/timestamp/linux_generic.hpp>
#elif defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
    #include <hpx/hardware/timestamp/linux_generic.hpp>
#elif defined(__ppc__) || defined(__ppc) || defined(__powerpc__)
    #include <hpx/hardware/timestamp/linux_generic.hpp>
#elif defined(__s390x__)
    #include <hpx/hardware/timestamp/linux_generic.hpp>
#elif defined(__bgq__)
    #include <hpx/hardware/timestamp/bgq.hpp>
#elif defined(__riscv)
    #include <hpx/hardware/timestamp/linux_riscv_64.hpp>
#else
    #error Unsupported platform.
#endif
// clang-format on
