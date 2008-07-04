//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BARRIER_JUN_23_2008_0530PM)
#define HPX_LCOS_BARRIER_JUN_23_2008_0530PM

#include <boost/detail/atomic_count.hpp>
#include <boost/lockfree/fifo.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    /// \class barrier barrier hpx/lcos/barrier.hpp
    ///
    /// A barrier can be used to synchronize a specific number of threads, 
    /// blocking all of the entering threads until all of the threads have 
    /// entered the barrier.
    ///
    /// \note   A \a barrier is not a LCO in the sense that it has a global id
    ///         or can be triggered using the action (parcel) mechanism. It
    ///         is just a low level synchronization primitive allowing to 
    ///         synchronize \a px_threads.
    class barrier 
    {
    public:
        barrier(int number_of_threads)
          : number_of_threads_(number_of_threads), count_(0)
        {
        }

        /// The function \a wait will block the first number entering threads
        /// (as given by the constructor parameter \a number_of_threads), 
        /// releasing all waiting threads as soon as the last \a px_thread
        /// entered this function.
        void wait(threadmanager::px_thread_self& self)
        {
            if (++count_ < number_of_threads_) {
                queue_.enqueue(self.get_thread_id());
                self.yield(threadmanager::suspended);
            }
            else {
                thread_id_type id = 0;
                while (--count_ > 0 && queue_.dequeue(&id)) 
                    threadmanager::set_state(self, id, threadmanager::pending);
            }
        }

    private:
        std::size_t number_of_threads_;
        boost::detail::atomic_count count_;
        boost::lockfree::fifo<thread_id_type> queue_;
    };

}}

#endif

