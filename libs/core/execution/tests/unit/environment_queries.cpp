//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/testing.hpp>

#include "algorithm_test_utils.hpp"

#include <exception>
#include <type_traits>
#include <utility>

bool set_value_sched_called = false;
bool set_value_delegatee_sched_called = false;
bool set_value_allocator_called = false;
bool set_value_stop_token_called = false;

namespace ex = hpx::execution::experimental;

namespace mylib {

#if defined(HPX_HAVE_STDEXEC)
    using sched = my_namespace::my_scheduler;
#else
    struct sched
    {
    };
#endif

#if defined(HPX_HAVE_STDEXEC)
    using sched_env_t = ex::prop<ex::get_scheduler_t, sched>;
#else
    using sched_env_t = ex::make_env_t<ex::get_scheduler_t, sched>;
#endif

#if defined(HPX_HAVE_STDEXEC)
    using delegatee_sched = my_namespace::my_scheduler_template<0>;
#else
    struct delegatee_sched
    {
    };
#endif

#if defined(HPX_HAVE_STDEXEC)
    using delegatee_sched_env_t = ex::env<sched_env_t,
        ex::prop<ex::get_delegatee_scheduler_t, delegatee_sched>>;
#else
    using delegatee_sched_env_t = ex::make_env_t<ex::get_delegatee_scheduler_t,
        delegatee_sched, sched_env_t>;
#endif

    struct allocator
    {
    };

#if defined(HPX_HAVE_STDEXEC)
    using allocator_env_t = ex::env<delegatee_sched_env_t,
        ex::prop<ex::get_allocator_t, allocator>>;
#else
    using allocator_env_t =
        ex::make_env_t<ex::get_allocator_t, allocator, delegatee_sched_env_t>;
#endif

    struct stop_token
    {
#if defined(HPX_HAVE_STDEXEC)
        // TODO: Find out the correct type for this alias.
        // Based on:
        // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/
        // p2300r10.html#design-cancellation-summary
        template <typename T>
        using callback_type = void;

        bool stop_requested() const noexcept
        {
            return false;
        };
        bool stop_possible() const noexcept
        {
            return true;
        };

        friend bool operator==([[maybe_unused]] stop_token const& a,
            [[maybe_unused]] stop_token const& b)
        {
            return false;
        }
#endif
    };

#if defined(HPX_HAVE_STDEXEC)
    // clang-format off
    using stop_token_env_t = ex::env<allocator_env_t,
        ex::prop<ex::get_stop_token_t, stop_token>>;
    // clang-format on
#else
    using stop_token_env_t =
        ex::make_env_t<ex::get_stop_token_t, stop_token, allocator_env_t>;
#endif

    struct receiver_1
    {
#if defined(HPX_HAVE_STDEXEC)
        using receiver_concept = ex::receiver_t;
#endif
        friend void tag_invoke(ex::set_stopped_t, receiver_1&&) noexcept {}

        friend void tag_invoke(
            ex::set_error_t, receiver_1&&, std::exception_ptr) noexcept
        {
        }

        friend void tag_invoke(ex::set_value_t, receiver_1&&, sched) noexcept
        {
            set_value_sched_called = true;
        }

        friend void tag_invoke(
            ex::set_value_t, receiver_1&&, delegatee_sched) noexcept
        {
            set_value_delegatee_sched_called = true;
        }

        friend void tag_invoke(
            ex::set_value_t, receiver_1&&, allocator) noexcept
        {
            set_value_allocator_called = true;
        }

        friend void tag_invoke(
            ex::set_value_t, receiver_1&&, stop_token) noexcept
        {
            set_value_stop_token_called = true;
        }

        friend auto tag_invoke(ex::get_env_t, receiver_1) noexcept
        {
#if defined(HPX_HAVE_STDEXEC)
            auto sched_env = ex::prop(ex::get_scheduler_t{}, sched());
#else
            auto sched_env = ex::make_env<ex::get_scheduler_t>(sched());
#endif
            static_assert(std::is_same_v<decltype(sched_env), sched_env_t>,
                "must return sched_env");
#if defined(HPX_HAVE_STDEXEC)
            auto delegatee_sched_env = ex::env(std::move(sched_env),
                ex::prop(ex::get_delegatee_scheduler_t{}, delegatee_sched()));
#else
            auto delegatee_sched_env =
                ex::make_env<ex::get_delegatee_scheduler_t>(
                    delegatee_sched(), sched_env);

#endif
            static_assert(std::is_same_v<decltype(delegatee_sched_env),
                              delegatee_sched_env_t>,
                "must return delegatee_sched_env");
#if defined(HPX_HAVE_STDEXEC)
            auto allocator_env = ex::env(std::move(delegatee_sched_env),
                ex::prop(ex::get_allocator_t{}, allocator()));
#else
            auto allocator_env = ex::make_env<ex::get_allocator_t>(
                allocator(), delegatee_sched_env);
#endif
            static_assert(
                std::is_same_v<decltype(allocator_env), allocator_env_t>,
                "must return allocator_env");
#if defined(HPX_HAVE_STDEXEC)
            auto stop_token_env = ex::env(std::move(allocator_env),
                ex::prop(ex::get_stop_token_t{}, stop_token()));
#else
            auto stop_token_env =
                ex::make_env<ex::get_stop_token_t>(stop_token(), allocator_env);
#endif
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
