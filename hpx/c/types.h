/*=============================================================================
    Copyright (c) 2007-2010 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#if !defined(HPX_C_TYPES_OCT_02_2009_1224PM)
#define HPX_C_TYPES_OCT_02_2009_1224PM

#if defined(__cplusplus)

#include <boost/shared_ptr.hpp>
#include <hpx/runtime/naming/name.hpp>

///////////////////////////////////////////////////////////////////////////////
// GID: global unique identifier 
typedef hpx::naming::id_type gid;

#else // #if defined(__cplusplus)

///////////////////////////////////////////////////////////////////////////////
// GID: global unique identifier (we know it's as big as two pointers)
struct gid 
{
    void *gid_;
    void *refcnt_; 
};

#endif // #if defined(__cplusplus)

#endif

