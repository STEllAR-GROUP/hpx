////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_UTIL_DETAIL_YIELD_WHILE_HPP
#define HPX_UTIL_DETAIL_YIELD_WHILE_HPP

#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/detail/yield_k.hpp>

#include <cstddef>

namespace hpx { namespace util {
    template <typename Predicate>
    inline void yield_while(Predicate && predicate,
        const char *thread_name = nullptr,
        hpx::threads::thread_state_enum p = hpx::threads::pending_boost,
        bool allow_timed_suspension = true)
    {
        if (allow_timed_suspension)
        {
            for (std::size_t k = 0; predicate(); ++k)
            {
                detail::yield_k(k, thread_name, p);
            }
        }
        else
        {
            for (std::size_t k = 0; predicate(); ++k)
            {
                detail::yield_k(k & 31/*k % 32*/, thread_name, p);
            }
        }
    }
}}

#endif
