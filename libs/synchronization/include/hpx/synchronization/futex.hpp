//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/timing/steady_clock.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstdint>

namespace hpx { namespace synchronization {
    // forward declaration of implementation details
    namespace detail {
        struct thread_entry;
    }
    /// The futex class is a low level synchronization primitve
    /// modeled after the linux futex user space synchronization facility
    /// Each futex maintains its own queue of blocked threads. The futex
    /// supports timed wait and a wakeup mechanism to wake up at most one thread
    struct HPX_EXPORT futex
    {
        futex() noexcept;

        ~futex() noexcept;

        threads::thread_state_ex_enum wait(char const* reason =
            "hpx::synchronization::futex::wait");

        threads::thread_state_ex_enum wait_until(
            util::steady_time_point const& abs_time, char const* reason =
            "hpx::synchronization::futex::wait_until");

        threads::thread_state_ex_enum wait_for(
            util::steady_duration const& rel_time, char const* reason =
            "hpx::synchronization::futex::wait_until")
        {
            return wait_until(rel_time.from_now(), reason);
        }

        // notify all waiting threads...
        void notify_all();
        // returns true if there are threads left blocking...
        bool notify_one();

        // abort all waiting threads
        void abort_all();
        // returns true if there are threads left blocking...
        bool abort_one();

    private:
        friend struct detail::thread_entry;
        using mutex_type = hpx::lcos::local::spinlock;

        mutex_type mtx_;
        detail::thread_entry* thread_head_;
        std::size_t num_references_;
        std::size_t epoch_;
    };

}}
