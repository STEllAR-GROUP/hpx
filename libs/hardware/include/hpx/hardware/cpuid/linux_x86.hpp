////////////////////////////////////////////////////////////////////////////////
//  Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
//  Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AC3F9C65_08DE_418F_9937_D9898E808454)
#define HPX_AC3F9C65_08DE_418F_9937_D9898E808454

#include <cstdint>

namespace hpx { namespace util { namespace hardware
{

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
    __asm__ __volatile__ (
        "cpuid ;\n"
        : "=a" (cpuinfo[cpuid_register::eax]),
          "=b" (cpuinfo[cpuid_register::ebx]),
          "=c" (cpuinfo[cpuid_register::ecx]),
          "=d" (cpuinfo[cpuid_register::edx])
        : "a" (eax)
        :);
}

void cpuidex(std::uint32_t (&cpuinfo)[4], std::uint32_t eax,
             std::uint32_t ecx)
{
    __asm__ __volatile__ (
        "cpuid ;\n"
        : "=a" (cpuinfo[cpuid_register::eax]),
          "=b" (cpuinfo[cpuid_register::ebx]),
          "=c" (cpuinfo[cpuid_register::ecx]),
          "=d" (cpuinfo[cpuid_register::edx])
        : "a" (eax), "c" (ecx)
        :);
}

}}}

#endif // HPX_AC3F9C65_08DE_418F_9937_D9898E808454

