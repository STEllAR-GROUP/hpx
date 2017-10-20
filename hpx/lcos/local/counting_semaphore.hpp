//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_COUNTING_SEMAPHORE_OCT_16_2008_1007AM)
#define HPX_LCOS_COUNTING_SEMAPHORE_OCT_16_2008_1007AM

#include <hpx/config.hpp>
#include <hpx/lcos/local/detail/counting_semaphore.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <cstdint>
#include <mutex>
#include <utility>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    /// A semaphore is a protected variable (an entity storing a value) or
    /// abstract data type (an entity grouping several variables that may or
    /// may not be numerical) which constitutes the classic method for
    /// restricting access to shared resources, such as shared memory, in a
    /// multiprogramming environment. Semaphores exist in many variants, though
    /// usually the term refers to a counting semaphore, since a binary
    /// semaphore is better known as a mutex. A counting semaphore is a counter
    /// for a set of available resources, rather than a locked/unlocked flag of
    /// a single resource. It was invented by Edsger Dijkstra. Semaphores are
    /// the classic solution to preventing race conditions in the dining
    /// philosophers problem, although they do not prevent resource deadlocks.
    ///
    /// Counting semaphores can be used for synchronizing multiple threads as
    /// well: one thread waiting for several other threads to touch (signal)
    /// the semaphore, or several threads waiting for one other thread to touch
    /// this semaphore.
    template <typename Mutex = hpx::lcos::local::spinlock, int N = 0>
    class counting_semaphore_var
    {
    private:
        typedef Mutex mutex_type;

    public:
        /// \brief Construct a new counting semaphore
        ///
        /// \param value    [in] The initial value of the internal semaphore
        ///                 lock count. Normally this value should be zero
        ///                 (which is the default), values greater than zero
        ///                 are equivalent to the same number of signals pre-
        ///                 set, and negative values are equivalent to the
        ///                 same number of waits pre-set.
        counting_semaphore_var(std::int64_t value = N)
          : sem_(value)
        {}

        /// \brief Wait for the semaphore to be signaled
        ///
        /// \param count    [in] The value by which the internal lock count will
        ///                 be decremented. At the same time this is the minimum
        ///                 value of the lock count at which the thread is not
        ///                 yielded.
        void wait(std::int64_t count = 1)
        {
            std::unique_lock<mutex_type> l(mtx_);
            sem_.wait(l, count);
        }

        /// \brief Try to wait for the semaphore to be signaled
        ///
        /// \param count    [in] The value by which the internal lock count will
        ///                 be decremented. At the same time this is the minimum
        ///                 value of the lock count at which the thread is not
        ///                 yielded.
        ///
        /// \returns        The function returns true if the calling thread was
        ///                 able to acquire the requested amount of credits.
        ///                 The function returns false if not sufficient credits
        ///                 are available at this point in time.
        bool try_wait(std::int64_t count = 1)
        {
            std::unique_lock<mutex_type> l(mtx_);
            return sem_.try_wait(l, count);
        }

        /// \brief Signal the semaphore
        void signal(std::int64_t count = 1)
        {
            std::unique_lock<mutex_type> l(mtx_);
            sem_.signal(std::move(l), count);
        }

        std::int64_t signal_all()
        {
            std::unique_lock<mutex_type> l(mtx_);
            return sem_.signal_all(std::move(l));
        }

    private:
        mutable mutex_type mtx_;
        detail::counting_semaphore sem_;
    };

    typedef counting_semaphore_var<> counting_semaphore;
}}}

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif

#endif

