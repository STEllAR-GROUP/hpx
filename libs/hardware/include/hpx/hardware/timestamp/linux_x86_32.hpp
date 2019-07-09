////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_1CA2FF5A_2757_440C_8D2D_240A48191E63)
#define HPX_1CA2FF5A_2757_440C_8D2D_240A48191E63

#include <cstdint>

namespace hpx { namespace util { namespace hardware
{

inline std::uint64_t timestamp()
{
    std::uint64_t r = 0;

    #if defined(HPX_HAVE_RDTSCP)
        __asm__ __volatile__ (
            "rdtscp ;\n"
            : "=A" (r)
            :
            : "%ecx");
    #elif defined(HPX_HAVE_RDTSC)
        __asm__ __volatile__ (
            "movl %%ebx, %%edi ;\n"
            "xorl %%eax, %%eax ;\n"
            "cpuid ;\n"
            "rdtsc ;\n"
            "movl %%edi, %%ebx ;\n"
            : "=A" (r)
            :
            : "%edi", "%ecx");
    #endif

    return r;
}

}}}

#endif // HPX_1CA2FF5A_2757_440C_8D2D_240A48191E63

