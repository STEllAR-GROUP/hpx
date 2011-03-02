////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_C488937C_79BE_4333_B4FF_DAFAE73E5AAF)
#define HPX_C488937C_79BE_4333_B4FF_DAFAE73E5AAF

#include <hpx/util/default_malloc.hpp>

#include <vector>

namespace hpx { namespace memory
{

template <typename T>
struct default_vector
{
    typedef std::vector<T, typename memory::default_allocator<T>::type> type; 
};

}}

#endif // HPX_C488937C_79BE_4333_B4FF_DAFAE73E5AAF

