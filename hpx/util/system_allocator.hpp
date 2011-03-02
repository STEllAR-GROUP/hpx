////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5E7FA4BF_4907_4757_A15F_97BDF229807C)
#define HPX_5E7FA4BF_4907_4757_A15F_97BDF229807C

#include <hpx/util/allocator.hpp>
#include <boost/cstdlib.hpp>

namespace hpx { namespace memory
{

// conforms to the Boost.Pool allocator interface
struct system
{
    typedef std::size_t size_type;    
    typedef std::ptrdiff_t difference_type;       
 
    static char* malloc(size_type s)
    {
        return reinterpret_cast<char*>(::malloc(s));
    }
    
    static void* void_malloc(size_type s)
    {
        return ::malloc(s);
    }
    
    template <typename T> 
    static T* object_malloc(size_type s)
    {
        return reinterpret_cast<T*>(::malloc(s * sizeof(T)));
    }

    template <typename T>  
    static T* realloc(T* p, size_type s)
    {
        return ::realloc(p, s);
    }

    template <typename T>  
    static void free(T* p)
    {
        ::free(reinterpret_cast<void*>(p));
    }
};

template <typename T>
struct system_allocator
{
    typedef memory::allocator<memory::system, T> type;
};

}}

#endif // HPX_5E7FA4BF_4907_4757_A15F_97BDF229807C

