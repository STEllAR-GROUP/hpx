//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution/queries/get_allocator.hpp>
#include <hpx/execution/queries/get_delegatee_scheduler.hpp>
#include <hpx/execution/queries/get_scheduler.hpp>
#include <hpx/execution/queries/get_stop_token.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <type_traits>

bool set_value_sched_called = false;
bool set_value_delegatee_sched_called = false;
bool set_value_allocator_called = false;
bool set_value_stop_token_called = false;

namespace ex = hpx::execution::experimental;

namespace mylib {

    struct sched
    {
    };

    using sched_env_t = ex::make_env_t<ex::get_scheduler_t, sched>;

    struct delegatee_sched
    {
    };

    using delegatee_sched_env_t = ex::make_env_t<ex::get_delegatee_scheduler_t,
        delegatee_sched, sched_env_t>;

    struct allocator
    {
    };

    using allocator_env_t =
        ex::make_env_t<ex::get_allocator_t, allocator, delegatee_sched_env_t>;

    struct stop_token
    {
    };

    using stop_token_env_t =
        ex::make_env_t<ex::get_stop_token_t, stop_token, allocator_env_t>;

    struct receiver_1
    {
        friend void tag_invoke(ex::set_stopped_t, receiver_1&&) noexcept {}

        friend void tag_invoke(
            ex::set_error_t, receiver_1&&, std::exception_ptr) noexcept
        {
        }

        friend void tag_invoke(ex::set_value_t, receiver_1&&, sched)
        {
            set_value_sched_called = true;
        }

        friend void tag_invoke(ex::set_value_t, receiver_1&&, delegatee_sched)
        {
            set_value_delegatee_sched_called = true;
        }

        friend void tag_invoke(ex::set_value_t, receiver_1&&, allocator)
        {
            set_value_allocator_called = true;
        }

        friend void tag_invoke(ex::set_value_t, receiver_1&&, stop_token)
        {
            set_value_stop_token_called = true;
        }

        friend constexpr auto tag_invoke(ex::get_env_t, receiver_1) noexcept
        {
            auto sched_env = ex::make_env<ex::get_scheduler_t>(sched());
            static_assert(std::is_same_v<decltype(sched_env), sched_env_t>,
                "must return sched_env");

            auto delegatee_sched_env =
                ex::make_env<ex::get_delegatee_scheduler_t>(
                    delegatee_sched(), sched_env);
            static_assert(std::is_same_v<decltype(delegatee_sched_env),
                              delegatee_sched_env_t>,
                "must return delegatee_sched_env");

            auto allocator_env = ex::make_env<ex::get_allocator_t>(
                allocator(), delegatee_sched_env);
            static_assert(
                std::is_same_v<decltype(allocator_env), allocator_env_t>,
                "must return allocator_env");

            auto stop_token_env =
                ex::make_env<ex::get_stop_token_t>(stop_token(), allocator_env);
            static_assert(
                std::is_same_v<decltype(stop_token_env), stop_token_env_t>,
                "must return stop_token_env");

            return stop_token_env;
        }
    };
}    // namespace mylib

int main()
{
    {
        mylib::receiver_1 rcv;
        auto env = ex::get_env(rcv);

        auto sched = ex::get_scheduler(env);
        static_assert(std::is_same_v<decltype(sched), mylib::sched>,
            "must return mylib::sched");

        auto delegatee_sched = ex::get_delegatee_scheduler(env);
        static_assert(
            std::is_same_v<decltype(delegatee_sched), mylib::delegatee_sched>,
            "must return mylib::delegatee_sched");

        auto allocator = ex::get_allocator(env);
        static_assert(std::is_same_v<decltype(allocator), mylib::allocator>,
            "must return mylib::allocator");

        auto stop_token = ex::get_stop_token(env);
        static_assert(std::is_same_v<decltype(stop_token), mylib::stop_token>,
            "must return mylib::stop_token");
    }
    {
        mylib::receiver_1 rcv;
        auto os = ex::connect(ex::get_scheduler(), rcv);
        ex::start(os);
        HPX_TEST(set_value_sched_called);
    }
    {
        mylib::receiver_1 rcv;
        auto os = ex::connect(ex::get_delegatee_scheduler(), rcv);
        ex::start(os);
        HPX_TEST(set_value_delegatee_sched_called);
    }
    {
        mylib::receiver_1 rcv;
        auto os = ex::connect(ex::get_allocator(), rcv);
        ex::start(os);
        HPX_TEST(set_value_allocator_called);
    }
    {
        mylib::receiver_1 rcv;
        auto os = ex::connect(ex::get_stop_token(), rcv);
        ex::start(os);
        HPX_TEST(set_value_stop_token_called);
    }

    return hpx::util::report_errors();
}
