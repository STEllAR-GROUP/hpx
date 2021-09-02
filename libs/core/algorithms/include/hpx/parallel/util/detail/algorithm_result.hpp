//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/executors/execution_policy_fwd.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/type_support/unused.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename T>
    struct algorithm_result_impl
    {
        // The return type of the initiating function.
        using type = T;

        // Obtain initiating function's return type.
        static constexpr type get()
        {
            return T();
        }

        template <typename T_>
        static constexpr type get(T_&& t)
        {
            return std::forward<T_>(t);
        }

        template <typename T_>
        static type get(hpx::future<T_>&& t)
        {
            return t.get();
        }
    };

    template <typename ExPolicy>
    struct algorithm_result_impl<ExPolicy, void>
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

    template <typename T>
    struct algorithm_result_impl<hpx::execution::sequenced_task_policy, T>
    {
        // The return type of the initiating function.
        using type = hpx::future<T>;

        // Obtain initiating function's return type.
        static type get(T&& t)
        {
            return hpx::make_ready_future(std::move(t));
        }

        static type get(hpx::future<T>&& t)
        {
            return std::move(t);
        }
    };

    template <>
    struct algorithm_result_impl<hpx::execution::sequenced_task_policy, void>
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

        static type get(hpx::future<void>&& t)
        {
            return std::move(t);
        }

        template <typename T>
        static type get(hpx::future<T>&& t)
        {
            return hpx::future<void>(std::move(t));
        }
    };

    template <typename T>
    struct algorithm_result_impl<hpx::execution::parallel_task_policy, T>
    {
        // The return type of the initiating function.
        using type = hpx::future<T>;

        // Obtain initiating function's return type.
        static type get(T&& t)
        {
            return hpx::make_ready_future(std::move(t));
        }

        static type get(hpx::future<T>&& t)
        {
            return std::move(t);
        }
    };

    template <>
    struct algorithm_result_impl<hpx::execution::parallel_task_policy, void>
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

        static type get(hpx::future<void>&& t)
        {
            return std::move(t);
        }

        template <typename T>
        static type get(hpx::future<T>&& t)
        {
            return hpx::future<void>(std::move(t));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename Parameters, typename T>
    struct algorithm_result_impl<
        hpx::execution::sequenced_task_policy_shim<Executor, Parameters>, T>
      : algorithm_result_impl<hpx::execution::sequenced_task_policy, T>
    {
    };

    template <typename Executor, typename Parameters>
    struct algorithm_result_impl<
        hpx::execution::sequenced_task_policy_shim<Executor, Parameters>, void>
      : algorithm_result_impl<hpx::execution::sequenced_task_policy, void>
    {
    };

    template <typename Executor, typename Parameters, typename T>
    struct algorithm_result_impl<
        hpx::execution::parallel_task_policy_shim<Executor, Parameters>, T>
      : algorithm_result_impl<hpx::execution::parallel_task_policy, T>
    {
    };

    template <typename Executor, typename Parameters>
    struct algorithm_result_impl<
        hpx::execution::parallel_task_policy_shim<Executor, Parameters>, void>
      : algorithm_result_impl<hpx::execution::parallel_task_policy, void>
    {
    };

#if defined(HPX_HAVE_DATAPAR)
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct algorithm_result_impl<hpx::execution::simd_task_policy, T>
      : algorithm_result_impl<hpx::execution::sequenced_task_policy, T>
    {
    };

    template <>
    struct algorithm_result_impl<hpx::execution::simd_task_policy, void>
      : algorithm_result_impl<hpx::execution::sequenced_task_policy, void>
    {
    };

    template <typename Executor, typename Parameters, typename T>
    struct algorithm_result_impl<
        hpx::execution::simd_task_policy_shim<Executor, Parameters>, T>
      : algorithm_result_impl<hpx::execution::sequenced_task_policy, T>
    {
    };

    template <typename Executor, typename Parameters>
    struct algorithm_result_impl<
        hpx::execution::simd_task_policy_shim<Executor, Parameters>, void>
      : algorithm_result_impl<hpx::execution::sequenced_task_policy, void>
    {
    };

    template <typename T>
    struct algorithm_result_impl<hpx::execution::par_simd_task_policy, T>
      : algorithm_result_impl<hpx::execution::parallel_task_policy, T>
    {
    };

    template <>
    struct algorithm_result_impl<hpx::execution::par_simd_task_policy, void>
      : algorithm_result_impl<hpx::execution::parallel_task_policy, void>
    {
    };

    template <typename Executor, typename Parameters, typename T>
    struct algorithm_result_impl<
        hpx::execution::par_simd_task_policy_shim<Executor, Parameters>, T>
      : algorithm_result_impl<hpx::execution::parallel_task_policy, T>
    {
    };

    template <typename Executor, typename Parameters>
    struct algorithm_result_impl<
        hpx::execution::par_simd_task_policy_shim<Executor, Parameters>, void>
      : algorithm_result_impl<hpx::execution::parallel_task_policy, void>
    {
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename T = void>
    struct algorithm_result
      : algorithm_result_impl<typename std::decay<ExPolicy>::type, T>
    {
        static_assert(!std::is_lvalue_reference<T>::value,
            "T shouldn't be a lvalue reference");
    };

    template <typename ExPolicy, typename T = void>
    using algorithm_result_t = typename algorithm_result<ExPolicy, T>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename U, typename Conv,
        HPX_CONCEPT_REQUIRES_(hpx::is_invocable_v<Conv, U>)>
    constexpr typename hpx::util::invoke_result<Conv, U>::type
    convert_to_result(U&& val, Conv&& conv)
    {
        return HPX_INVOKE(conv, val);
    }

    template <typename U, typename Conv,
        HPX_CONCEPT_REQUIRES_(hpx::is_invocable_v<Conv, U>)>
    hpx::future<typename hpx::util::invoke_result<Conv, U>::type>
    convert_to_result(hpx::future<U>&& f, Conv&& conv)
    {
        using result_type = typename hpx::util::invoke_result<Conv, U>::type;

        return hpx::make_future<result_type>(
            std::move(f), std::forward<Conv>(conv));
    }
}}}}    // namespace hpx::parallel::util::detail
