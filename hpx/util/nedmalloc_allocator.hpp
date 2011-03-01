////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_3C74DADC_8743_4902_993B_C3323BF2A863)
#define HPX_3C74DADC_8743_4902_993B_C3323BF2A863

#include <hpx/util/allocator.hpp>
#include <hpx_nedmalloc/nedmalloc.h>

namespace hpx { namespace memory
{

// conforms to the Boost.Pool allocator interface
struct nedmalloc
{
    typedef std::size_t size_type;    
    typedef std::ptrdiff_t difference_type;       
 
    static char* malloc(size_type s)
    {
        return reinterpret_cast<char*>(nedalloc::nedmalloc(s));
    }
 
    static void* void_malloc(size_type s)
    {
        return nedalloc::nedmalloc(s);
    }
    
    template <typename T> 
    static T* object_malloc(size_type s)
    {
        return reinterpret_cast<T*>(nedalloc::nedmalloc(s * sizeof(T)));
    }

    template <typename T>  
    static void free(T* p)
    {
        nedalloc::nedfree(reinterpret_cast<void*>(p));
    }
};

template <typename T>
struct nedmalloc_allocator
{
    typedef memory::allocator<memory::nedmalloc, T> type;
};

}}

#endif // HPX_3C74DADC_8743_4902_993B_C3323BF2A863

