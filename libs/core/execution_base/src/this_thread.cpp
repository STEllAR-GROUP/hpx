//  Copyright (c) 2013-2019 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/execution_base/agent_base.hpp>
#include <hpx/execution_base/context_base.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

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

namespace hpx::execution_base {

    namespace {

        struct default_context final : execution_base::context_base
        {
            resource_base const& resource() const noexcept override
            {
                return resource_;
            }
            resource_base resource_;
        };

        struct default_agent : execution_base::agent_base
        {
            default_agent();

            [[nodiscard]] std::string description() const override
            {
                return hpx::util::format("{}", id_);
            }

            [[nodiscard]] default_context const& context()
                const noexcept override
            {
                return context_;
            }

            void yield(char const* desc) override;
            void yield_k(std::size_t k, char const* desc) override;
            void suspend(char const* desc) override;
            void resume(hpx::threads::thread_priority priority,
                char const* desc) override;
            void abort(char const* desc) override;
            void sleep_for(hpx::chrono::steady_duration const& sleep_duration,
                char const* desc) override;
            void sleep_until(hpx::chrono::steady_time_point const& sleep_time,
                char const* desc) override;

        private:
            bool running_;
            bool aborted_;
            std::thread::id id_;
            std::mutex mtx_;
            std::condition_variable suspend_cv_;
            std::condition_variable resume_cv_;

            default_context context_;
        };

        default_agent::default_agent()
          : running_(true)
          , aborted_(false)
          , id_(std::this_thread::get_id())
        {
        }

        void default_agent::yield(char const* /* desc */)
        {
#if defined(HPX_WINDOWS)
            Sleep(0);
#else
            sched_yield();
#endif
        }

        void default_agent::yield_k(std::size_t k, char const* /* desc */)
        {
            if (k < 4)    //-V112
            {
            }
            else if (k < 16)
            {
                HPX_SMT_PAUSE;
            }
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

        void default_agent::suspend(char const* /* desc */)
        {
            std::unique_lock<std::mutex> l(mtx_);
            HPX_ASSERT(running_);

            running_ = false;
            resume_cv_.notify_all();

            while (!running_)
            {
                suspend_cv_.wait(l);    //-V1089
            }

            if (aborted_)
            {
                HPX_THROW_EXCEPTION(hpx::error::yield_aborted, "suspend",
                    "std::thread({}) aborted (yield returned wait_abort)", id_);
            }
        }

        void default_agent::resume(hpx::threads::thread_priority, char const*)
        {
            {
                std::unique_lock<std::mutex> l(mtx_);
                while (running_)
                {
                    resume_cv_.wait(l);    //-V1089
                }
                running_ = true;
            }
            suspend_cv_.notify_one();
        }

        void default_agent::abort(char const* /* desc */)
        {
            {
                std::unique_lock<std::mutex> l(mtx_);
                while (running_)
                {
                    resume_cv_.wait(l);    //-V1089
                }
                running_ = true;
                aborted_ = true;
            }
            suspend_cv_.notify_one();
        }

        void default_agent::sleep_for(
            hpx::chrono::steady_duration const& sleep_duration,
            char const* /* desc */)
        {
            std::this_thread::sleep_for(sleep_duration.value());
        }

        void default_agent::sleep_until(
            hpx::chrono::steady_time_point const& sleep_time,
            char const* /* desc */)
        {
            std::this_thread::sleep_until(sleep_time.value());
        }
    }    // namespace

    namespace detail {

        [[nodiscard]] agent_base& get_default_agent()
        {
            static thread_local default_agent agent;
            return agent;
        }
    }    // namespace detail

    namespace this_thread {

        namespace detail {

            struct agent_storage
            {
                agent_storage()
                  : impl_(&hpx::execution_base::detail::get_default_agent())
                {
                }

                agent_base* set(agent_base* context) noexcept
                {
                    std::swap(context, impl_);
                    return context;
                }

                agent_base* impl_;
            };

            [[nodiscard]] agent_storage* get_agent_storage()
            {
                static thread_local agent_storage storage;
                return &storage;
            }
        }    // namespace detail

        reset_agent::reset_agent(
            detail::agent_storage* storage, agent_base& impl)
          : storage_(storage)
          , old_(storage_->set(&impl))
        {
        }

        reset_agent::reset_agent(agent_base& impl)
          : reset_agent(detail::get_agent_storage(), impl)
        {
        }

        reset_agent::~reset_agent()
        {
            storage_->set(old_);
        }

        [[nodiscard]] agent_ref agent()
        {
            return agent_ref(detail::get_agent_storage()->impl_);
        }

        void yield(char const* desc)
        {
            agent().yield(desc);
        }

        void yield_k(std::size_t k, char const* desc)
        {
            agent().yield_k(k, desc);
        }

        void suspend(char const* desc)
        {
            agent().suspend(desc);
        }
    }    // namespace this_thread
}    // namespace hpx::execution_base
