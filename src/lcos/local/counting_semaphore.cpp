//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/stringstream.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    counting_semaphore::~counting_semaphore()
    {
        mutex_type::scoped_lock l(mtx_);

        if (!queue_.empty())
        {
            LERR_(fatal)
                << "lcos::counting_semaphore::~counting_semaphore:"
                   " queue is not empty, aborting threads";

            while (!queue_.empty())
            {
                threads::thread_id_type id = queue_.front().id_;
                queue_.front().id_ = 0;
                queue_.pop_front();

                // we know that the id is actually the pointer to the thread
                threads::thread_data* thrd = static_cast<threads::thread_data*>(id);
                LERR_(fatal)
                        << "lcos::counting_semaphore::~counting_semaphore:"
                        << " pending thread: "
                        << get_thread_state_name(thrd->get_state())
                        << "(" << id << "): " << thrd->get_description();

                // forcefully abort thread, do not throw
                error_code ec;
                threads::set_thread_state(id, threads::pending,
                    threads::wait_abort, threads::thread_priority_normal, ec);
                if (ec)
                {
                    LERR_(fatal)
                        << "lcos::counting_semaphore::~counting_semaphore:"
                        << " could not abort thread: "
                        << get_thread_state_name(thrd->get_state())
                        << "(" << id << "): " << thrd->get_description();
                }
            }
        }
    }

    void counting_semaphore::wait_locked(boost::int64_t count,
        mutex_type::scoped_lock& l)
    {
        while (value_ < count)
        {
            // we need to get the self anew for each round as it might
            // get executed in a different thread from the previous one
            threads::thread_self& self = threads::get_self();
            threads::thread_id_type id = self.get_thread_id();

            threads::set_thread_lco_description(id, "lcos::counting_semaphore");

            queue_entry e(id);
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
                strm << "thread(" << id << ", "
                     << threads::get_thread_description(id)
                     << ") aborted (yield returned wait_abort)";
                HPX_THROW_EXCEPTION(yield_aborted,
                    "lcos::counting_semaphore::wait",
                    hpx::util::osstream_get_string(strm));
                return;
            }
        }

        value_ -= count;
    }

    void counting_semaphore::signal_locked(boost::int64_t count,
        mutex_type::scoped_lock& l)
    {
        value_ += count;
        if (value_ >= 0)
        {
            // release all threads, they will figure out between themselves
            // which one gets released from wait above
#if BOOST_VERSION < 103600
            // slist::swap has a bug in Boost 1.35.0
            while (!queue_.empty())
            {
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
            while (!queue.empty())
            {
                threads::thread_id_type id = queue.front().id_;
                if (HPX_UNLIKELY(!id)) {
                    HPX_THROW_EXCEPTION(null_thread_id,
                        "counting_semaphore::signal_locked",
                        "NULL thread id encountered");
                }
                queue.front().id_ = 0;
                queue.pop_front();
                threads::set_thread_lco_description(id);
                threads::set_thread_state(id, threads::pending);
            }
#endif
        }
    }
}}}

