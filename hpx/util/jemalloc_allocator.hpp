////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5E7FA4BF_4907_4757_A15F_97BDF229807C)
#define HPX_5E7FA4BF_4907_4757_A15F_97BDF229807C

#include <hpx/util/allocator.hpp>
#include <hpx_jemalloc/jemalloc.h>

namespace hpx { namespace memory
{

// conforms to the Boost.Pool allocator interface
struct jemalloc
{
    typedef std::size_t size_type;    
    typedef std::ptrdiff_t difference_type;       
 
    static char* malloc(size_type s)
    {
        return reinterpret_cast<char*>(HPX_JEMALLOC_malloc(s));
    }
    
    static void* void_malloc(size_type s)
    {
        return HPX_JEMALLOC_malloc(s);
    }
    
    template <typename T> 
    static T* object_malloc(size_type s)
    {
        return reinterpret_cast<T*>(HPX_JEMALLOC_malloc(s * sizeof(T)));
    }

    template <typename T>  
    static void free(T* p)
    {
        HPX_JEMALLOC_free(reinterpret_cast<void*>(p));
    }
};

template <typename T>
struct jemalloc_allocator
{
    typedef memory::allocator<memory::jemalloc, T> type;
};

}}

#endif // HPX_5E7FA4BF_4907_4757_A15F_97BDF229807C

