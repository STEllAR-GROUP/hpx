//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_COUNTING_SEMAPHORE_OCT_16_2008_1007AM)
#define HPX_LCOS_COUNTING_SEMAPHORE_OCT_16_2008_1007AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/util/unlock_lock.hpp>

#include <boost/assert.hpp>
#include <boost/thread.hpp>
#include <boost/lockfree/fifo.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    /// \class counting_semaphore counting_semaphore hpx/lcos/counting_semaphore.hpp
    ///
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
    class counting_semaphore
    {
    private:
//         typedef hpx::lcos::mutex mutex_type;
        typedef boost::mutex mutex_type;

    public:
        /// \brief Construct a new counting semaphore
        ///
        /// \param value    [in] The initial value of the internal semaphore 
        ///                 lock count. Normally this value should be zero
        ///                 (which is the default), values greater than zero
        ///                 are equivalent to the same number of signals pre-
        ///                 set, and negative values are equivalent to the
        ///                 same number of waits pre-set.
        counting_semaphore(long value = 0)
          : value_(value)
        {}

        ~counting_semaphore()
        {
            BOOST_ASSERT(queue_.empty());   // queue has to be empty
        }

        /// \brief Wait for the semaphore to be signaled
        ///
        /// \param count    [in] The value by which the internal lock count will 
        ///                 be decremented. At the same time this is the minimum 
        ///                 value of the lock count at which the thread is not 
        ///                 yielded.
        void wait()
        {
            mutex_type::scoped_lock l(mtx_);

            while (0 == value_) {     // allow for higher priority threads
                // we need to get the self anew for each round as it might
                // get executed in a different thread from the previous one
                threads::thread_self& self = threads::get_self();
                queue_.enqueue(self.get_thread_id());

                util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                self.yield(threads::suspended);
            }

            --value_;
        }

        /// \brief Signal the semaphore
        ///
        /// 
        void signal()
        {
            mutex_type::scoped_lock l(mtx_);

            threads::thread_id_type id = 0;
            if (queue_.dequeue(&id)) 
                threads::set_thread_state(id, threads::pending);

            ++value_;
        }

    private:
        mutex_type mtx_;
        long value_;
        boost::lockfree::fifo<threads::thread_id_type> queue_;
    };

}}

#endif
