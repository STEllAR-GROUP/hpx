//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This is partially taken from: http://www.garret.ru/threadalloc/readme.html

#if !defined(HPX_UTIL_ALLOCATOR_DELETER_AUG_08_2018_1047AM)
#define HPX_UTIL_ALLOCATOR_DELETER_AUG_08_2018_1047AM

#include <hpx/config.hpp>

#include <memory>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Allocator>
    struct allocator_deleter
    {
        template <typename SharedState>
        void operator()(SharedState* state)
        {
            using traits = std::allocator_traits<Allocator>;
            traits::deallocate(alloc_, state, 1);
        }

        Allocator alloc_;
    };
}}

#endif

