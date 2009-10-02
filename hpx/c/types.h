/*=============================================================================
    Copyright (c) 2007-2009 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#if !defined(HPX_C_TYPES_OCT_02_2009_1224PM)
#define HPX_C_TYPES_OCT_02_2009_1224PM

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
# ifdef __hpux    // HP-UX has a vaguely nice <stdint.h> in a non-standard location
#  include <inttypes.h>
# elif defined(__FreeBSD__) || defined(__IBMCPP__) || defined(_AIX)
#  include <inttypes.h>
# else
#  include <stdint.h>
# endif
#else
# if !defined(_MSC_VER)
typedef unsigned long long uint64_t;
# else
typedef unsigned __int64 uint64_t;
# endif
#endif

///////////////////////////////////////////////////////////////////////////////
#if defined(__cplusplus)
extern "C" {
#endif

// GID: global unique identifier
struct gid 
{
    uint64_t id_msb_;
    uint64_t id_lsb_;
};

#if defined(__cplusplus)
}
#endif

#endif

