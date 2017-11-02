//  Copyright (c) 2017 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/detail/scheduling_loop.hpp>

#include <atomic>
#include <cstdint>

namespace hpx { namespace threads { namespace detail
{
    // TODO: Find the right way to do this.
    static std::atomic<std::int64_t> background_thread_count;

    std::int64_t get_background_thread_count()
    {
        return background_thread_count.load();
    }

    void increment_background_thread_count()
    {
        ++background_thread_count;
    }

    void decrement_background_thread_count()
    {
        --background_thread_count;
    }
}}}

