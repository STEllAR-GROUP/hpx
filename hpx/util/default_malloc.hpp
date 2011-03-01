////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_76CF5A02_404F_4174_BCA2_6DEA15D4D981)
#define HPX_76CF5A02_404F_4174_BCA2_6DEA15D4D981

#if defined(BOOST_WINDOWS)
    #include <hpx/util/nedmalloc_allocator.hpp>
#else
    #include <hpx/util/jemalloc_allocator.hpp>
#endif

namespace hpx { namespace memory
{

#if defined(BOOST_WINDOWS)
    typedef hpx::memory::nedmalloc default_malloc; 
#else
    typedef hpx::memory::jemalloc default_malloc; 
#endif

template <typename T>
struct default_allocator
{
    typedef memory::allocator<memory::default_malloc, T> type;
};

}}

#endif // HPX_76CF5A02_404F_4174_BCA2_6DEA15D4D981

