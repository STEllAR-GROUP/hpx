//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/synchronization/detail/counting_semaphore.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local {
    // A semaphore is a protected variable (an entity storing a value) or
    // abstract data type (an entity grouping several variables that may or
    // may not be numerical) which constitutes the classic method for
    // restricting access to shared resources, such as shared memory, in a
    // multiprogramming environment. Semaphores exist in many variants, though
    // usually the term refers to a counting semaphore, since a binary
    // semaphore is better known as a mutex. A counting semaphore is a counter
    // for a set of available resources, rather than a locked/unlocked flag of
    // a single resource. It was invented by Edsger Dijkstra. Semaphores are
    // the classic solution to preventing race conditions in the dining
    // philosophers problem, although they do not prevent resource deadlocks.
    //
    // Counting semaphores can be used for synchronizing multiple threads as
    // well: one thread waiting for several other threads to touch (signal)
    // the semaphore, or several threads waiting for one other thread to touch
    // this semaphore.
    template <std::ptrdiff_t LeastMaxValue = PTRDIFF_MAX,
        typename Mutex = hpx::lcos::local::spinlock>
    class cpp20_counting_semaphore
    {
    public:
        HPX_NON_COPYABLE(cpp20_counting_semaphore);

    protected:
        using mutex_type = Mutex;

    public:
        // Returns The maximum value of counter. This value is greater than or
        // equal to LeastMaxValue.
        static constexpr std::ptrdiff_t(max)() noexcept
        {
            return LeastMaxValue;
        }

        // \brief Construct a new counting semaphore
        //
        // \param value    [in] The initial value of the internal semaphore
        //                 lock count. Normally this value should be zero
        //                 (which is the default), values greater than zero
        //                 are equivalent to the same number of signals pre-
        //                 set, and negative values are equivalent to the
        //                 same number of waits pre-set.
        //
        //  Preconditions   value >= 0 is true, and value <= max() is true.
        //  Effects         Initializes counter with desired.
        //  Throws          Nothing.
        explicit cpp20_counting_semaphore(std::ptrdiff_t value)
          : sem_(value)
        {
        }

        ~cpp20_counting_semaphore() = default;

        // Preconditions:   update >= 0 is true, and update <= max() - counter
        //                  is true.
        // Effects:         Atomically execute counter += update. Then, unblocks
        //                  any threads that are waiting for counter to be
        //                  greater than zero.
        // Synchronization: Strongly happens before invocations of try_acquire
        //                  that observe the result of the effects.
        // Throws:          system_error when an exception is required
        //                  ([thread.req.exception]).
        // Error conditions: Any of the error conditions allowed for mutex
        //                  types ([thread.mutex.requirements.mutex]).
        void release(std::ptrdiff_t update = 1)
        {
            std::unique_lock<mutex_type> l(mtx_);
            sem_.signal(std::move(l), update);
        }

        // Effects:         Attempts to atomically decrement counter if it is
        //                  positive, without blocking. If counter is not
        //                  decremented, there is no effect and try_acquire
        //                  immediately returns. An implementation may fail to
        //                  decrement counter even if it is positive. [ Note:
        //                  This spurious failure is normally uncommon, but
        //                  allows interesting implementations based on a simple
        //                  compare and exchange ([atomics]). - end note ]
        //                  An implementation should ensure that try_acquire
        //                  does not consistently return false in the absence
        //                  of contending semaphore operations.
        // Returns:         true if counter was decremented, otherwise false.
        bool try_acquire() noexcept
        {
            std::unique_lock<mutex_type> l(mtx_);
            return sem_.try_acquire(l);
        }

        // Effects:         Repeatedly performs the following steps, in order:
        //                    - Evaluates try_acquire. If the result is true,
        //                      returns.
        //                    - Blocks on *this until counter is greater than
        //                      zero.
        // Throws:          system_error when an exception is required
        //                  ([thread.req.exception]).
        // Error conditions: Any of the error conditions allowed for mutex
        //                  types ([thread.mutex.requirements.mutex]).
        void acquire()
        {
            std::unique_lock<mutex_type> l(mtx_);
            sem_.wait(l, 1);
        }

        // Effects:         Repeatedly performs the following steps, in order:
        //                    - Evaluates try_acquire(). If the result is true,
        //                      returns true.
        //                    - Blocks on *this until counter is greater than
        //                      zero or until the timeout expires. If it is
        //                      unblocked by the timeout expiring, returns false.
        //                  The timeout expires ([thread.req.timing]) when the
        //                  current time is after abs_time (for
        //                  try_acquire_until) or when at least rel_time has
        //                  passed from the start of the function (for
        //                  try_acquire_for).
        // Throws:          Timeout - related exceptions ([thread.req.timing]),
        //                  or system_error when a non - timeout - related
        //                  exception is required ([thread.req.exception]).
        // Error conditions: Any of the error conditions allowed for mutex types
        //                  ([thread.mutex.requirements.mutex]).
        bool try_acquire_until(util::steady_time_point const& abs_time)
        {
            std::unique_lock<mutex_type> l(mtx_);
            return sem_.wait_until(l, abs_time, 1);
        }

        bool try_acquire_for(util::steady_duration const& rel_time)
        {
            return try_acquire_until(rel_time.from_now());
        }

    protected:
        mutable mutex_type mtx_;
        detail::counting_semaphore sem_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = hpx::lcos::local::spinlock>
    class cpp20_binary_semaphore : public cpp20_counting_semaphore<1, Mutex>
    {
    public:
        HPX_NON_COPYABLE(cpp20_binary_semaphore);

    public:
        cpp20_binary_semaphore(std::ptrdiff_t value = 1)
          : cpp20_counting_semaphore<1, Mutex>(value)
        {
        }

        ~cpp20_binary_semaphore() = default;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Mutex = hpx::lcos::local::spinlock, int N = 0>
    class counting_semaphore_var : cpp20_counting_semaphore<PTRDIFF_MAX, Mutex>
    {
    private:
        using mutex_type = Mutex;

    public:
        // \brief Construct a new counting semaphore
        //
        // \param value    [in] The initial value of the internal semaphore
        //                 lock count. Normally this value should be zero
        //                 (which is the default), values greater than zero
        //                 are equivalent to the same number of signals pre-
        //                 set, and negative values are equivalent to the
        //                 same number of waits pre-set.
        //
        //  Preconditions   value >= 0 is true, and value <= max() is true.
        //  Effects         Initializes counter with desired.
        //  Throws          Nothing.
        explicit counting_semaphore_var(std::ptrdiff_t value = N)
          : cpp20_counting_semaphore<PTRDIFF_MAX, Mutex>(value)
        {
        }

        counting_semaphore_var(counting_semaphore_var const&) = delete;
        counting_semaphore_var& operator=(
            counting_semaphore_var const&) = delete;

        // \brief Wait for the semaphore to be signaled
        //
        // \param count    [in] The value by which the internal lock count will
        //                 be decremented. At the same time this is the minimum
        //                 value of the lock count at which the thread is not
        //                 yielded.
        void wait(std::ptrdiff_t count = 1)
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            this->sem_.wait(l, count);
        }

        // \brief Try to wait for the semaphore to be signaled
        //
        // \param count    [in] The value by which the internal lock count will
        //                 be decremented. At the same time this is the minimum
        //                 value of the lock count at which the thread is not
        //                 yielded.
        //
        // \returns        The function returns true if the calling thread was
        //                 able to acquire the requested amount of credits.
        //                 The function returns false if not sufficient credits
        //                 are available at this point in time.
        bool try_wait(std::ptrdiff_t count = 1)
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            return this->sem_.try_wait(l, count);
        }

        /// \brief Signal the semaphore
        void signal(std::ptrdiff_t count = 1)
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            this->sem_.signal(std::move(l), count);
        }

        std::ptrdiff_t signal_all()
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            return this->sem_.signal_all(std::move(l));
        }
    };

    typedef counting_semaphore_var<> counting_semaphore;
}}}    // namespace hpx::lcos::local

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif
