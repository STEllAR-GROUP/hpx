//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BARRIER_JUN_23_2008_0530PM)
#define HPX_LCOS_BARRIER_JUN_23_2008_0530PM

#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/thread/locks.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    /// A barrier can be used to synchronize a specific number of threads,
    /// blocking all of the entering threads until all of the threads have
    /// entered the barrier.
    ///
    /// \note   A \a barrier is not a LCO in the sense that it has no global id
    ///         and it can't be triggered using the action (parcel) mechanism.
    ///         It is just a low level synchronization primitive allowing to
    ///         synchronize a given number of \a threads.
    class barrier
    {
    private:
        typedef lcos::local::spinlock mutex_type;

    public:
        barrier(std::size_t number_of_threads)
          : number_of_threads_(number_of_threads)
        {}

        /// The function \a wait will block the number of entering \a threads
        /// (as given by the constructor parameter \a number_of_threads),
        /// releasing all waiting threads as soon as the last \a thread
        /// entered this function.
        void wait()
        {
            boost::unique_lock<mutex_type> l(mtx_);
            if (cond_.size(l) < number_of_threads_-1) {
                cond_.wait(std::move(l), "barrier::wait");
            }
            else {
                // release the threads
                cond_.notify_all(std::move(l));
            }
        }

    private:
        std::size_t const number_of_threads_;
        mutable mutex_type mtx_;
        local::detail::condition_variable cond_;
    };
}}}

#endif

