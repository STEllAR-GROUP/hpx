//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/just.hpp>
#include <hpx/execution/algorithms/sync_wait.hpp>
#include <hpx/execution/algorithms/then.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/executors/execution_policy_fwd.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/type_support/unused.hpp>

#include <type_traits>
#include <utility>

namespace hpx::parallel::util::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename T, typename Enable = void>
    struct algorithm_result_impl;

    template <typename ExPolicy, typename T>
    struct algorithm_result_impl<ExPolicy, T,
        std::enable_if_t<!hpx::is_async_execution_policy_v<ExPolicy> &&
            !hpx::execution_policy_has_scheduler_executor_v<ExPolicy>>>
    {
        // The return type of the initiating function.
        using type = T;

        // Obtain initiating function's return type.
        static constexpr type get() noexcept(
            std::is_nothrow_default_constructible_v<T>)
        {
            return T();
        }

        template <typename T_>
        static constexpr auto get(T_&& t)
        {
            return HPX_FORWARD(T_, t);
        }

        template <typename T_>
        static auto get(hpx::future<T_>&& t)
        {
            return t.get();
        }
    };

    template <typename ExPolicy>
    struct algorithm_result_impl<ExPolicy, void,
        std::enable_if_t<!hpx::is_async_execution_policy_v<ExPolicy> &&
            !hpx::execution_policy_has_scheduler_executor_v<ExPolicy>>>
    {
        // The return type of the initiating function.
        using type = void;

        // Obtain initiating function's return type.
        static constexpr void get() noexcept {}

        static constexpr void get(hpx::util::unused_type) noexcept {}

        static void get(hpx::future<void>&& t)
        {
            t.get();
        }

        template <typename T>
        static void get(hpx::future<T>&& t)
        {
            t.get();
        }
    };

    template <typename ExPolicy, typename T>
    struct algorithm_result_impl<ExPolicy, T,
        std::enable_if_t<hpx::is_async_execution_policy_v<ExPolicy> &&
            !hpx::execution_policy_has_scheduler_executor_v<ExPolicy>>>
    {
        // The return type of the initiating function.
        using type = hpx::future<T>;

        // Obtain initiating function's return type.
        static type get(T&& t)
        {
            return hpx::make_ready_future(HPX_MOVE(t));
        }

        static type get(hpx::future<T>&& t)
        {
            return HPX_MOVE(t);
        }
    };

    template <typename ExPolicy>
    struct algorithm_result_impl<ExPolicy, void,
        std::enable_if_t<hpx::is_async_execution_policy_v<ExPolicy> &&
            !hpx::execution_policy_has_scheduler_executor_v<ExPolicy>>>
    {
        // The return type of the initiating function.
        using type = hpx::future<void>;

        // Obtain initiating function's return type.
        static type get()
        {
            return hpx::make_ready_future();
        }

        static type get(hpx::util::unused_type)
        {
            return hpx::make_ready_future();
        }

        static type get(hpx::future<void>&& t) noexcept
        {
            return HPX_MOVE(t);
        }

        template <typename T>
        static type get(hpx::future<T>&& t) noexcept
        {
            return hpx::future<void>(HPX_MOVE(t));
        }
    };

    template <typename ExPolicy, typename T>
    struct algorithm_result_impl<ExPolicy, T,
        std::enable_if_t<!hpx::is_async_execution_policy_v<ExPolicy> &&
            hpx::execution_policy_has_scheduler_executor_v<ExPolicy>>>
    {
        // The return type of the initiating function.
        using type = T;

        static constexpr type get() noexcept(
            std::is_nothrow_default_constructible_v<T>)
        {
            return T();
        }

        template <typename T_>
        static constexpr auto get(T_&& t)
        {
            namespace ex = hpx::execution::experimental;
            if constexpr (ex::is_sender_v<T_>)
            {
                namespace tt = hpx::this_thread::experimental;
                auto result = tt::sync_wait(HPX_FORWARD(T_, t));
                if constexpr (hpx::tuple_size_v<
                                  std::decay_t<decltype(*result)>> == 0)
                {
                    return;
                }
                else
                {
                    return hpx::get<0>(*result);
                }
            }
            else
            {
                return HPX_FORWARD(T_, t);
            }
        }
    };

    template <typename ExPolicy>
    struct algorithm_result_impl<ExPolicy, void,
        std::enable_if_t<!hpx::is_async_execution_policy_v<ExPolicy> &&
            hpx::execution_policy_has_scheduler_executor_v<ExPolicy>>>
    {
        // The return type of the initiating function.
        using type = void;

        static constexpr type get() noexcept {}

        template <typename T>
        static constexpr auto get(T&& t)
        {
            namespace ex = hpx::execution::experimental;
            if constexpr (ex::is_sender_v<T>)
            {
                namespace tt = hpx::this_thread::experimental;
                tt::sync_wait(HPX_FORWARD(T, t));
            }
        }
    };

    template <typename ExPolicy, typename T>
    struct algorithm_result_impl<ExPolicy, T,
        std::enable_if_t<hpx::is_async_execution_policy_v<ExPolicy> &&
            hpx::execution_policy_has_scheduler_executor_v<ExPolicy>>>
    {
        // The return type of the initiating function.
        using type = T;

        template <typename T_>
        static constexpr auto get(T_&& t)
        {
            namespace ex = hpx::execution::experimental;
            if constexpr (ex::is_sender_v<T_>)
            {
                return HPX_FORWARD(T_, t);
            }
            else
            {
                return ex::just(HPX_FORWARD(T_, t));
            }
        }
    };

    template <typename ExPolicy>
    struct algorithm_result_impl<ExPolicy, void,
        std::enable_if_t<hpx::is_async_execution_policy_v<ExPolicy> &&
            hpx::execution_policy_has_scheduler_executor_v<ExPolicy>>>
    {
        // The return type of the initiating function.
        using type = void;

        template <typename T_>
        static constexpr auto get(T_&& t)
        {
            namespace ex = hpx::execution::experimental;
            if constexpr (ex::is_sender_v<T_>)
            {
                return HPX_FORWARD(T_, t);
            }
            else
            {
                return ex::just(HPX_FORWARD(T_, t));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename T = void>
    struct algorithm_result : algorithm_result_impl<std::decay_t<ExPolicy>, T>
    {
        static_assert(!std::is_lvalue_reference_v<T>,
            "T shouldn't be a lvalue reference");
    };

    template <typename ExPolicy, typename T = void>
    using algorithm_result_t = typename algorithm_result<ExPolicy, T>::type;

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off
    template <typename U, typename Conv,
        HPX_CONCEPT_REQUIRES_(
           !hpx::execution::experimental::is_sender_v<U> &&
            hpx::is_invocable_v<Conv, U>
        )>
    // clang-format on
    constexpr hpx::util::invoke_result_t<Conv, U> convert_to_result(
        U&& val, Conv&& conv)
    {
        return HPX_INVOKE(conv, val);
    }

    // clang-format off
    template <typename Sender, typename Conv,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_sender_v<Sender>
        )>
    // clang-format on
    constexpr decltype(auto) convert_to_result(Sender&& sender, Conv&& conv)
    {
        return hpx::execution::experimental::then(HPX_FORWARD(Sender, sender),
            [conv = HPX_FORWARD(Conv, conv)](auto&& value) mutable {
                return HPX_INVOKE(conv, HPX_FORWARD(decltype(value), value));
            });
    }

    template <typename U, typename Conv,
        HPX_CONCEPT_REQUIRES_(hpx::is_invocable_v<Conv, U>)>
    hpx::future<hpx::util::invoke_result_t<Conv, U>> convert_to_result(
        hpx::future<U>&& f, Conv&& conv)
    {
        using result_type = hpx::util::invoke_result_t<Conv, U>;

        return hpx::make_future<result_type>(
            HPX_MOVE(f), HPX_FORWARD(Conv, conv));
    }
}    // namespace hpx::parallel::util::detail
