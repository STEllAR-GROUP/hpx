////////////////////////////////////////////////////////////////////////////////
//  Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
//  Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_WINDOWS)

#include <cstdint>

#include <intrin.h>

namespace hpx { namespace util { namespace hardware {

    struct cpuid_register
    {
        enum info
        {
            eax = 0,
            ebx = 1,
            ecx = 2,
            edx = 3
        };
    };

    void cpuid(std::uint32_t (&cpuinfo)[4], std::uint32_t eax)
    {
        ::__cpuid(cpuinfo, eax);
    }

    void cpuidex(
        std::uint32_t (&cpuinfo)[4], std::uint32_t eax, std::uint32_t ecx)
    {
        ::__cpuidex(cpuinfo, eax, ecx);
    }

}}}    // namespace hpx::util::hardware

#endif
