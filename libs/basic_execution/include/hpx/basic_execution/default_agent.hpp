//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_BASIC_EXECUTION_DEFAULT_AGENT_HPP
#define HPX_BASIC_EXECUTION_DEFAULT_AGENT_HPP

#include <hpx/basic_execution/agent_base.hpp>
#include <hpx/basic_execution/default_context.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>

namespace hpx { namespace basic_execution {
    struct HPX_EXPORT default_agent : hpx::basic_execution::agent_base
    {
        default_agent();

        std::string description() const override;

        default_context const& context() const override
        {
            return context_;
        }

        void yield(char const* desc) override;
        void yield_k(std::size_t k, char const* desc) override;
        void yield_to(agent_base& agent, char const* desc) override;
        void suspend(char const* desc) override;
        void resume(char const* desc) override;
        void abort(char const* desc) override;
        void sleep_for(hpx::util::steady_duration const& sleep_duration,
            char const* desc) override;
        void sleep_until(hpx::util::steady_time_point const& sleep_time,
            char const* desc) override;

    private:
        bool running_;
        bool aborted_;
        std::thread::id id_;
        std::mutex mtx_;
        std::condition_variable suspend_cv_;
        std::condition_variable resume_cv_;

        static default_context context_;
    };
}}

#endif
