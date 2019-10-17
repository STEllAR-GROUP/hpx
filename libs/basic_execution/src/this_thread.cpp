//  Copyright (c) 2013-2019 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/basic_execution/agent_base.hpp>
#include <hpx/basic_execution/context_base.hpp>
#include <hpx/basic_execution/default_agent.hpp>
#include <hpx/basic_execution/this_thread.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace hpx { namespace basic_execution { namespace this_thread {
    namespace detail {
        agent_base& get_default_agent()
        {
            static thread_local default_agent agent;
            return agent;
        }

        struct agent_storage
        {
            agent_storage()
              : impl_(&get_default_agent())
            {
            }

            agent_base* set(agent_base* context) noexcept
            {
                std::swap(context, impl_);
                return context;
            }

            agent_base* impl_;
        };

        agent_storage* get_agent_storage()
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

    hpx::basic_execution::agent_ref agent()
    {
        return hpx::basic_execution::agent_ref(
            detail::get_agent_storage()->impl_);
    }

    void yield(char const* desc)
    {
        agent().yield(desc);
    }

    void yield_k(std::size_t k, char const* desc)
    {
        agent().yield_k(k, desc);
    }

    void yield_to(agent_ref agnt, char const* desc)
    {
        agent().yield_to(agnt, desc);
    }

    void suspend(char const* desc)
    {
        agent().suspend(desc);
    }
}}}       // namespace hpx::basic_execution::this_thread
