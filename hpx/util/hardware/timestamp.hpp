////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E19D2797_B133_4BDB_8E25_2DD9BDE7E093)
#define HPX_E19D2797_B133_4BDB_8E25_2DD9BDE7E093

#include <hpx/config.hpp>

#if defined(HPX_MSVC)
  #include <hpx/util/hardware/timestamp/msvc.hpp>
#elif defined(__amd64__) || defined(__amd64)    \
   || defined(__x86_64__) || defined(__x86_64)  \
   || defined(_M_X64)
    #if defined(HPX_HAVE_RDTSC) || defined(HPX_HAVE_RDTSCP)
        #include <hpx/util/hardware/timestamp/linux_x86_64.hpp>
    #else
        #include <hpx/util/hardware/timestamp/linux_generic.hpp>
    #endif
#elif defined(i386) || defined(__i386__) || defined(__i486__)           \
   || defined(__i586__) || defined(__i686__) || defined(__i386)         \
   || defined(_M_IX86) || defined(__X86__) || defined(_X86_)            \
   || defined(__THW_INTEL__) || defined(__I86__) || defined(__INTEL__)
    #if defined(HPX_HAVE_RDTSC) || defined(HPX_HAVE_RDTSCP)
        #include <hpx/util/hardware/timestamp/linux_x86_32.hpp>
    #else
        #include <hpx/util/hardware/timestamp/linux_generic.hpp>
    #endif
#elif (defined(__ANDROID__) && defined(ANDROID)) || defined(__arm__)
    #include <hpx/util/hardware/timestamp/linux_generic.hpp>
#elif defined(__powerpc__)
    #include <hpx/util/hardware/timestamp/linux_generic.hpp>
#elif defined(__bgq__)
    #include <hpx/util/hardware/timestamp/bgq.hpp>
#else
    #error Unsupported platform.
#endif

#endif // HPX_E19D2797_B133_4BDB_8E25_2DD9BDE7E093


