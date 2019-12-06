//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  The algorithm was taken from http://locklessinc.com/articles/barriers/

#if !defined(HPX_LCOS_BARRIER_JUN_23_2008_0530PM)
#define HPX_LCOS_BARRIER_JUN_23_2008_0530PM

#include <hpx/config.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <climits>
#include <cstddef>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local {
    /// A barrier can be used to synchronize a specific number of threads,
    /// blocking all of the entering threads until all of the threads have
    /// entered the barrier.
    ///
    /// \note   A \a barrier is not a LCO in the sense that it has no global id
    ///         and it can't be triggered using the action (parcel) mechanism.
    ///         It is just a low level synchronization primitive allowing to
    ///         synchronize a given number of \a threads.
    class HPX_EXPORT barrier
    {
    private:
        typedef lcos::local::spinlock mutex_type;

        HPX_STATIC_CONSTEXPR std::size_t barrier_flag =
            static_cast<std::size_t>(1) << (CHAR_BIT * sizeof(std::size_t) - 1);

    public:
        barrier(std::size_t number_of_threads);
        ~barrier();

        /// The function \a wait will block the number of entering \a threads
        /// (as given by the constructor parameter \a number_of_threads),
        /// releasing all waiting threads as soon as the last \a thread
        /// entered this function.
        void wait();

        /// The function \a count_up will increase the number of \a threads
        /// to be waited in \a wait function.
        void count_up();

        /// The function \a reset will reset the number of \a threads
        /// as given by the function parameter \a number_of_threads.
        /// the newer coming \a threads executing the function
        /// \a wait will be waiting until \a total_ is equal to  \a barrier_flag.
        /// The last \a thread exiting the \a wait function will notify
        /// the newer \a threads waiting and the newer \a threads
        /// will get the reset \a number_of_threads_.
        /// The function \a reset can be executed while previous \a threads
        /// executing waiting after they have been waken up.
        /// Thus \a total_ can not be reset to \a barrier_flag which
        /// will break the comparison condition under the function \a wait.
        void reset(std::size_t number_of_threads);

    private:
        std::size_t number_of_threads_;
        std::size_t total_;

        mutable mutex_type mtx_;
        local::detail::condition_variable cond_;
    };
}}}    // namespace hpx::lcos::local

#include <hpx/config/warnings_suffix.hpp>

#endif
