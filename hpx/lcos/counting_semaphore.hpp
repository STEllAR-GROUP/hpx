//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_COUNTING_SEMAPHORE_OCT_16_2008_1007AM)
#define HPX_LCOS_COUNTING_SEMAPHORE_OCT_16_2008_1007AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>

#include <boost/assert.hpp>
#include <boost/intrusive/slist.hpp>

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
        struct tag {};
        typedef hpx::util::spinlock_pool<tag> mutex_type;

        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_entry(threads::thread_id_type id)
              : id_(id)
            {}

            threads::thread_id_type id_;
            hook_type slist_hook_;
        };

        typedef boost::intrusive::member_hook<
            queue_entry, 
            typename queue_entry::hook_type,
            &queue_entry::slist_hook_
        > slist_option_type;

        typedef boost::intrusive::slist<
            queue_entry, slist_option_type, 
            boost::intrusive::cache_last<true>, 
            boost::intrusive::constant_time_size<false>
        > queue_type;

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
        void wait(long count = 1)
        {
            mutex_type::scoped_lock l(this);

            while (value_ < count) {
                // we need to get the self anew for each round as it might
                // get executed in a different thread from the previous one
                threads::thread_self& self = threads::get_self();
                threads::thread_id_type id = self.get_thread_id();

                // mark the thread as suspended before adding to the queue
                reinterpret_cast<threads::thread*>(id)->
                    set_state(threads::marked_for_suspension);

                queue_entry f(id);
                queue_.push_back(f);

                util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                self.yield(threads::suspended);
            }

            value_ -= count;
        }

        /// \brief Signal the semaphore
        ///
        /// 
        void signal(long count = 1)
        {
            mutex_type::scoped_lock l(this);

            value_ += count;
            if (value_ >= 0) {
                // release all threads, they will figure out between themselves
                // which one gets released from wait above
                while (!queue_.empty()) {
                    queue_entry& e (queue_.front());
                    queue_.pop_front();
                    threads::set_thread_state(e.id_, threads::pending);
                }
            }
        }

    private:
        long value_;
        queue_type queue_;
    };

}}

#endif
