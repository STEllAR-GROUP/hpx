//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SLIDING_SEMAPHORE_AUG_25_2016_1028AM)
#define HPX_LCOS_SLIDING_SEMAPHORE_AUG_25_2016_1028AM

#include <hpx/config.hpp>
#include <hpx/lcos/local/detail/sliding_semaphore.hpp>
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
    /// Sliding semaphores can be used for synchronizing multiple threads as
    /// well: one thread waiting for several other threads to touch (signal)
    /// the semaphore, or several threads waiting for one other thread to touch
    /// this semaphore. The difference to a counting semaphore is that a
    /// sliding semaphore will not limit the number of threads which are
    /// allowed to proceed, but will make sure that the difference between
    /// the (arbitrary) number passed to set and wait does not exceed a given
    /// threshold.
    template <typename Mutex = hpx::lcos::local::spinlock>
    class sliding_semaphore_var
    {
    private:
        typedef Mutex mutex_type;

    public:
        /// \brief Construct a new sliding semaphore
        ///
        /// \param max_difference
        ///                 [in] The max difference between the upper limit
        ///                 (as set by wait()) and the lower limit (as set by
        ///                 signal()) which is allowed without suspending any
        ///                 thread calling wait().
        /// \param lower_limit  [in] The initial lower limit.
        sliding_semaphore_var(std::int64_t max_difference,
                std::int64_t lower_limit = 0)
          : sem_(max_difference, lower_limit)
        {}

        /// \brief Wait for the semaphore to be signaled
        ///
        /// \param upper_limit [in] The new upper limit.
        ///           The calling thread will be suspended if the difference
        ///           between this value and the largest lower_limit which was
        ///           set by signal() is larger than the max_difference.
        void wait(std::int64_t upper_limit)
        {
            std::unique_lock<mutex_type> l(mtx_);
            sem_.wait(l, upper_limit);
        }

        /// \brief Try to wait for the semaphore to be signaled
        ///
        /// \param upper_limit [in] The new upper limit.
        ///           The calling thread will be suspended if the difference
        ///           between this value and the largest lower_limit which was
        ///           set by signal() is larger than the max_difference.
        ///
        /// \returns  The function returns true if the calling thread
        ///           would not block if it was calling wait().
        bool try_wait(std::int64_t upper_limit = 1)
        {
            std::unique_lock<mutex_type> l(mtx_);
            return sem_.try_wait(l, upper_limit);
        }

        /// \brief Signal the semaphore
        ///
        /// \param lower_limit  [in] The new lower limit. This will update the
        ///             current lower limit of this semaphore. It will also
        ///             re-schedule all suspended threads for which their
        ///             associated upper limit is not larger than the lower
        ///             limit plus the max_difference.
        void signal(std::int64_t lower_limit)
        {
            std::unique_lock<mutex_type> l(mtx_);
            sem_.signal(std::move(l), lower_limit);
        }

        std::int64_t signal_all()
        {
            std::unique_lock<mutex_type> l(mtx_);
            return sem_.signal_all(std::move(l));
        }

    private:
        mutable mutex_type mtx_;
        detail::sliding_semaphore sem_;
    };

    typedef sliding_semaphore_var<> sliding_semaphore;
}}}

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif

#endif

