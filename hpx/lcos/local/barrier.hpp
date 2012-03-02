//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_barrier_JUN_23_2008_0530PM)
#define HPX_LCOS_barrier_JUN_23_2008_0530PM

#include <boost/intrusive/slist.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/stringstream.hpp>

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

        // define data structures needed for intrusive slist container used for
        // the queues
        struct barrier_queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            barrier_queue_entry(threads::thread_id_type id)
              : id_(id)
            {}

            threads::thread_id_type id_;
            hook_type slist_hook_;
        };

        typedef boost::intrusive::member_hook<
            barrier_queue_entry, barrier_queue_entry::hook_type,
            &barrier_queue_entry::slist_hook_
        > slist_option_type;

        typedef boost::intrusive::slist<
            barrier_queue_entry, slist_option_type,
            boost::intrusive::cache_last<true>,
            boost::intrusive::constant_time_size<false>
        > queue_type;

    public:
        barrier(std::size_t number_of_threads)
          : number_of_threads_(number_of_threads)
        {}

        ~barrier()
        {
            if (!queue_.empty()) {
                LERR_(fatal) << "~barrier: thread_queue is not empty, aborting threads";

                mutex_type::scoped_lock l(mtx_);
                while (!queue_.empty()) {
                    threads::thread_id_type id = queue_.front().id_;
                    queue_.front().id_ = 0;
                    queue_.pop_front();

                    // we know that the id is actually the pointer to the thread
                    threads::thread* thrd = static_cast<threads::thread*>(id);
                    LERR_(fatal) << "~barrier: pending thread: "
                            << get_thread_state_name(thrd->get_state())
                            << "(" << id << "): " << thrd->get_description();

                    // forcefully abort thread, do not throw
                    error_code ec;
                    threads::set_thread_state(id, threads::pending,
                        threads::wait_abort, threads::thread_priority_normal, ec);
                    if (ec) {
                        LERR_(fatal) << "~barrier: could not abort thread"
                            << get_thread_state_name(thrd->get_state())
                            << "(" << id << "): " << thrd->get_description();
                    }
                }
            }
        }

        /// The function \a wait will block the number of entering \a threads
        /// (as given by the constructor parameter \a number_of_threads),
        /// releasing all waiting threads as soon as the last \a thread
        /// entered this function.
        void wait()
        {
            threads::thread_self& self = threads::get_self();

            mutex_type::scoped_lock l(mtx_);
            if (queue_.size() < number_of_threads_-1) {
                threads::thread_id_type id = self.get_thread_id();

                threads::set_thread_lco_description(id, "lcos::barrier");

                barrier_queue_entry e(id);
                queue_.push_back(e);
                queue_type::const_iterator last = queue_.last();
                threads::thread_state_ex_enum statex;

                {
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                    statex = self.yield(threads::suspended);
                }

                if (e.id_)
                    queue_.erase(last);     // remove entry from queue

                if (statex == threads::wait_abort) {
                    hpx::util::osstream strm;
                    strm << "thread(" << id << ", " << threads::get_thread_description(id)
                          << ") aborted (yield returned wait_abort)";
                    HPX_THROW_EXCEPTION(yield_aborted, "barrier::wait",
                        hpx::util::osstream_get_string(strm));
                    return;
                }
            }
            else {
            // slist::swap has a bug in Boost 1.35.0
#if BOOST_VERSION < 103600
                // release the threads
                while (!queue_.empty()) {
                    threads::thread_id_type id = queue_.front().id_;
                    queue_.front().id_ = 0;
                    queue_.pop_front();
                    threads::set_thread_lco_description(id);
                    threads::set_thread_state(id, threads::pending);
                }
#else
                // swap the list
                queue_type queue;
                queue.swap(queue_);
                l.unlock();

                // release the threads
                while (!queue.empty()) {
                    threads::thread_id_type id = queue.front().id_;
                    queue.front().id_ = 0;
                    queue.pop_front();
                    threads::set_thread_lco_description(id);
                    threads::set_thread_state(id, threads::pending);
                }
#endif
            }
        }

    private:
        std::size_t const number_of_threads_;
        mutable mutex_type mtx_;
        queue_type queue_;
    };
}}}

#endif

