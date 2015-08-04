// (C) Copyright 2007-9 Anthony Williams
// Copyright (c) 2015 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADS_DETAIL_THREAD_GROUP_AUG_03_2015_0908PM)
#define HPX_THREADS_DETAIL_THREAD_GROUP_AUG_03_2015_0908PM

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/local/shared_mutex.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/util/move.hpp>

#include <boost/thread/lock_guard.hpp>

#include <list>
#include <algorithm>

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4251)
#endif

namespace hpx { namespace threads { namespace detail
{
    class thread_group
    {
    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(thread_group);

        typedef lcos::local::shared_mutex mutex_type;

    public:
        thread_group() {}

        ~thread_group()
        {
            for (thread* t: threads)
                delete t;
        }

    private:
        bool is_this_thread_in()
        {
            thread::id id = this_thread::get_id();
            boost::shared_lock<mutex_type> guard(mtx_);
            for (thread* t: threads)
            {
                if (t->get_id() == id)
                    return true;
            }
            return false;
        }

        bool is_thread_in(thread* thrd)
        {
            if (!thrd)
                return false;

            thread::id id = thrd->get_id();
            boost::shared_lock<mutex_type> guard(mtx_);
            for (thread* t: threads)
            {
                if (t->get_id() == id)
                    return true;
            }
            return false;
        }

    public:
        template<typename F>
        thread* create_thread(F && f)
        {
            boost::lock_guard<mutex_type> guard(mtx_);
            std::unique_ptr<thread> new_thread(new thread(std::forward<F>(f)));
            threads.push_back(new_thread.get());
            return new_thread.release();
        }

        void add_thread(thread* thrd)
        {
            if(thrd)
            {
                if(is_thread_in(thrd))
                {
                    HPX_THROW_EXCEPTION(
                        thread_resource_error, "thread_group::add_thread",
                        "resource_deadlock_would_occur: trying to add a "
                            "duplicated thread");
                    return;
                };

                boost::lock_guard<mutex_type> guard(mtx_);
                threads.push_back(thrd);
            }
        }

        void remove_thread(thread* thrd)
        {
            boost::lock_guard<mutex_type> guard(mtx_);
            std::list<thread*>::iterator const it =
                std::find(threads.begin(), threads.end(), thrd);

            if(it != threads.end())
                threads.erase(it);
        }

        void join_all()
        {
            if (is_this_thread_in())
            {
                HPX_THROW_EXCEPTION(
                    thread_resource_error, "thread_group::join_all",
                    "resource_deadlock_would_occur: trying joining itself");
                return;
            }

            boost::shared_lock<mutex_type> guard(mtx_);
            for (thread* t: threads)
            {
                if (t->joinable())
                    t->join();
            }
        }

        void interrupt_all()
        {
            boost::shared_lock<mutex_type> guard(mtx_);
            for (thread* t: threads)
            {
                t->interrupt();
            }
        }

        size_t size() const
        {
            boost::shared_lock<mutex_type> guard(mtx_);
            return threads.size();
        }

    private:
        std::list<thread*> threads;
        mutable mutex_type mtx_;
    };
}}}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif
