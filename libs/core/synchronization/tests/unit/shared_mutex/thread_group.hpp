// (C) Copyright 2007-9 Anthony Williams
// Copyright (c) 2015-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/modules/threading.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/shared_mutex.hpp>

#include <algorithm>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>

#ifdef HPX_MSVC
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace test {

    class thread_group
    {
    private:
        using mutex_type = hpx::shared_mutex;

    public:
        thread_group() {}

        thread_group(thread_group const&) = delete;
        thread_group& operator=(thread_group const&) = delete;

        ~thread_group()
        {
            for (hpx::thread const* t : threads)
                delete t;
        }

    private:
        bool is_this_thread_in() const
        {
            hpx::thread::id const id = hpx::this_thread::get_id();
            std::shared_lock<mutex_type> guard(mtx_);
            for (hpx::thread const* t : threads)
            {
                if (t->get_id() == id)
                    return true;
            }
            return false;
        }

        bool is_thread_in(hpx::thread const* thrd) const
        {
            if (!thrd)
                return false;

            hpx::thread::id const id = thrd->get_id();
            std::shared_lock<mutex_type> guard(mtx_);
            for (hpx::thread const* t : threads)
            {
                if (t->get_id() == id)
                    return true;
            }
            return false;
        }

    public:
        template <typename F>
        hpx::thread* create_thread(F&& f)
        {
            std::lock_guard<mutex_type> guard(mtx_);
            std::unique_ptr<hpx::thread> new_thread(
                new hpx::thread(HPX_FORWARD(F, f)));
            threads.push_back(new_thread.get());
            return new_thread.release();
        }

        void add_thread(hpx::thread* thrd)
        {
            if (thrd)
            {
                if (is_thread_in(thrd))
                {
                    HPX_THROW_EXCEPTION(hpx::error::thread_resource_error,
                        "thread_group::add_thread",
                        "resource_deadlock_would_occur: trying to add a "
                        "duplicated thread");
                };

                std::lock_guard<mutex_type> guard(mtx_);
                threads.push_back(thrd);
            }
        }

        void remove_thread(hpx::thread const* thrd)
        {
            std::lock_guard<mutex_type> guard(mtx_);
            std::list<hpx::thread*>::iterator const it =
                std::find(threads.begin(), threads.end(), thrd);

            if (it != threads.end())
                threads.erase(it);
        }

        void join_all() const
        {
            if (is_this_thread_in())
            {
                HPX_THROW_EXCEPTION(hpx::error::thread_resource_error,
                    "thread_group::join_all",
                    "resource_deadlock_would_occur: trying joining itself");
            }

            std::shared_lock<mutex_type> guard(mtx_);
            for (hpx::thread* t : threads)
            {
                if (t->joinable())
                    t->join();
            }
        }

        void interrupt_all() const
        {
            std::shared_lock<mutex_type> guard(mtx_);
            for (hpx::thread* t : threads)
            {
                t->interrupt();
            }
        }

        size_t size() const
        {
            std::shared_lock<mutex_type> guard(mtx_);
            return threads.size();
        }

    private:
        std::list<hpx::thread*> threads;
        mutable mutex_type mtx_;
    };
}    // namespace test

#ifdef HPX_MSVC
#pragma warning(pop)
#endif
