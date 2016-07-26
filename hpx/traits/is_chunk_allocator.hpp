//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_CHUNK_ALLOCATOR_MAR_24_2015)
#define HPX_TRAITS_IS_CHUNK_ALLOCATOR_MAR_24_2015

#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename A, typename Enable = void>
    struct is_chunk_allocator : std::false_type {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename A, typename Enable = void>
    struct default_chunk_size
    {
        static std::size_t call(A const &a) { return 0; }
    };

    template <typename A>
    struct default_chunk_size<
            A, typename std::enable_if<is_chunk_allocator<A>::value>::type
        >
    {
        static std::size_t call(A const &a)
        {
            return a.memory_pool_->chunk_size_;
        }
    };
}}

#endif
