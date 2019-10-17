//  Copyright (c) 2013-2019 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/basic_execution/default_agent.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/format.hpp>

#include <mutex>
#include <thread>

#if defined(HPX_WINDOWS)
#include <windows.h>
#else
#ifndef _AIX
#include <sched.h>
#else
// AIX's sched.h defines ::var which sometimes conflicts with Lambda's var
extern "C" int sched_yield(void);
#endif
#include <time.h>
#endif

namespace hpx { namespace basic_execution {
    default_context default_agent::context_;

    default_agent::default_agent()
      : running_(true)
      , aborted_(false)
      , id_(std::this_thread::get_id())
    {
    }

    std::string default_agent::description() const
    {
        return hpx::util::format("{}", id_);
    }

    void default_agent::yield(char const* desc)
    {
#if defined(HPX_SMT_PAUSE)
        HPX_SMT_PAUSE;
#else
#if defined(HPX_WINDOWS)
        Sleep(0);
#else
        sched_yield();
#endif
#endif
    }

    void default_agent::yield_k(std::size_t k, char const* desc)
    {
        if (k < 4)    //-V112
        {
        }
#if defined(HPX_SMT_PAUSE)
        else if (k < 16)
        {
            HPX_SMT_PAUSE;
        }
#endif
        else if (k < 32 || k & 1)    //-V112
        {
#if defined(HPX_WINDOWS)
            Sleep(0);
#else
            sched_yield();
#endif
        }
        else
        {
#if defined(HPX_WINDOWS)
            Sleep(1);
#else
            // g++ -Wextra warns on {} or {0}
            struct timespec rqtp = {0, 0};

            // POSIX says that timespec has tv_sec and tv_nsec
            // But it doesn't guarantee order or placement

            rqtp.tv_sec = 0;
            rqtp.tv_nsec = 1000;

            nanosleep(&rqtp, nullptr);
#endif
        }
    }

    void default_agent::yield_to(agent_base& agent, char const* desc)
    {
        if (agent.context() != context())
        {
            agent.resume("default_agent::yield_to");
            yield(desc);
            return;
        }
        HPX_ASSERT(dynamic_cast<default_agent*>(&agent));
        auto& agent_impl = static_cast<default_agent &>(agent);
        agent_impl.resume("default_agent::yield_to");
        yield(desc);
    }

    void default_agent::suspend(char const* desc)
    {
        std::unique_lock<std::mutex> l(mtx_);
        HPX_ASSERT(running_);

        running_ = false;
        resume_cv_.notify_all();

        while (!running_)
        {
            suspend_cv_.wait(l);
        }

        if (aborted_)
        {
            HPX_THROW_EXCEPTION(yield_aborted, "suspend",
                hpx::util::format(
                    "std::thread({}) aborted (yield returned wait_abort)",
                    id_));
        }
    }

    void default_agent::resume(char const* desc)
    {
        {
            std::unique_lock<std::mutex> l(mtx_);
            while (running_)
            {
                resume_cv_.wait(l);
            }
            running_ = true;
        }
        suspend_cv_.notify_one();
    }

    void default_agent::abort(char const* desc)
    {
        {
            std::unique_lock<std::mutex> l(mtx_);
            while (running_)
            {
                resume_cv_.wait(l);
            }
            running_ = true;
            aborted_ = true;
        }
        suspend_cv_.notify_one();
    }

    void default_agent::sleep_for(
        hpx::util::steady_duration const& sleep_duration, char const* desc)
    {
        std::this_thread::sleep_for(sleep_duration.value());
    }

    void default_agent::sleep_until(
        hpx::util::steady_time_point const& sleep_time, char const* desc)
    {
        std::this_thread::sleep_until(sleep_time.value());
    }
}}
