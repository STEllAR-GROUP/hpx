////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_7309AC21_1B92_4CC1_8ACC_CAA4E841DB3A)
#define HPX_7309AC21_1B92_4CC1_8ACC_CAA4E841DB3A

#if defined(__amd64__) || defined(__amd64) || defined(__x86_64__)   \
   || defined(__x86_64) || defined(_M_X64) || defined(i386)         \
   || defined(__i386__) || defined(__i486__) || defined(__i586__)   \
   || defined(__i686__) || defined(__i386) || defined(_M_IX86)      \
   || defined(__X86__) || defined(_X86_) || defined(__THW_INTEL__)  \
   || defined(__I86__) || defined(__INTEL__)
    #include <boost/cstdint.hpp>

    #include <hpx/util/hardware/bit_manipulation.hpp>

    #if defined(HPX_MSVC)
        #include <hpx/util/hardware/cpuid/msvc.hpp>
    #else
        #include <hpx/util/hardware/cpuid/linux_x86.hpp>
    #endif

    namespace hpx { namespace util { namespace hardware
    {

    typedef boost::uint32_t cpu_info [4];

    struct cpu_feature
    {
        enum info
        {
            clflush = 0,
            cx8     = 1,
            cx16    = 2,
            cmovcc  = 3,
            msr     = 4,
            rdtsc   = 5,
            rdtscp  = 6,
            mmx     = 7,
            sse     = 8,
            sse2    = 9,
            sse3    = 10,
            ssse3   = 11,
            sse4_1  = 12,
            sse4_2  = 13,
            avx     = 14,
            xop     = 15,
            fma     = 16
        };
    };

    struct cpuid_table_type
    {
        cpu_feature::info feature;
        char const* name;
        boost::uint32_t function;
        cpuid_register::info register_;
        boost::uint32_t bit;
    };

    const cpuid_table_type cpuid_table[] = {
        {cpu_feature::clflush, "clflush", 0x00000001, cpuid_register::edx, 19},
        {cpu_feature::cx8,     "cx8",     0x00000001, cpuid_register::edx, 8},
        {cpu_feature::cx16,    "cx16",    0x00000001, cpuid_register::ecx, 13},
        {cpu_feature::cmovcc,  "cmovcc",  0x00000001, cpuid_register::edx, 15},
        {cpu_feature::msr,     "msr",     0x00000001, cpuid_register::edx, 5},
        {cpu_feature::rdtsc,   "rdtsc",   0x00000001, cpuid_register::edx, 4},
        {cpu_feature::rdtscp,  "rdtscp",  0x80000001, cpuid_register::edx, 27},
        {cpu_feature::mmx,     "mmx",     0x00000001, cpuid_register::edx, 23},
        {cpu_feature::sse,     "sse",     0x00000001, cpuid_register::edx, 25},
        {cpu_feature::sse2,    "sse2",    0x00000001, cpuid_register::edx, 26},
        {cpu_feature::sse3,    "sse3",    0x00000001, cpuid_register::ecx, 0},
        {cpu_feature::ssse3,   "ssse3",   0x00000001, cpuid_register::ecx, 9},
        {cpu_feature::sse4_1,  "sse4.1",  0x00000001, cpuid_register::ecx, 19},
        {cpu_feature::sse4_2,  "sse4.2",  0x00000001, cpuid_register::ecx, 20},
        {cpu_feature::avx,     "avx",     0x00000001, cpuid_register::ecx, 28},
        {cpu_feature::xop,     "xop",     0x80000001, cpuid_register::edx, 11},
        {cpu_feature::fma,     "fma",     0x80000001, cpuid_register::edx, 16}
    };

    }}}
#else
    #error Unsupported platform.
#endif

#endif // HPX_7309AC21_1B92_4CC1_8ACC_CAA4E841DB3A

