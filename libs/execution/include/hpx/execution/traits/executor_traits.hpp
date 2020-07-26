//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/detected.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace lcos {
    template <typename R>
    class future;
}}    // namespace hpx::lcos

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    struct static_chunk_size;

    ///////////////////////////////////////////////////////////////////////////
    struct sequenced_execution_tag;
    struct parallel_execution_tag;
    struct unsequenced_execution_tag;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(post)
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(sync_execute)
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(async_execute)
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(then_execute)
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(bulk_sync_execute)
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(bulk_async_execute)
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(bulk_then_execute)
    }    // namespace detail

    template <typename T, typename Enable = void>
    struct has_post_member : detail::has_post<typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_sync_execute_member
      : detail::has_sync_execute<typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_async_execute_member
      : detail::has_async_execute<typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_then_execute_member
      : detail::has_then_execute<typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_sync_execute_member
      : detail::has_bulk_sync_execute<typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_async_execute_member
      : detail::has_bulk_async_execute<typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_then_execute_member
      : detail::has_bulk_then_execute<typename std::decay<T>::type>
    {
    };

#if defined(HPX_HAVE_CXX17_VARIABLE_TEMPLATES)
    template <typename T>
    constexpr bool has_post_member_v = has_post_member<T>::value;

    template <typename T>
    constexpr bool has_sync_execute_member_v =
        has_sync_execute_member<T>::value;

    template <typename T>
    constexpr bool has_async_execute_member_v =
        has_async_execute_member<T>::value;

    template <typename T>
    constexpr bool has_then_execute_member_v =
        has_then_execute_member<T>::value;

    template <typename T>
    constexpr bool has_bulk_sync_execute_member_v =
        has_bulk_sync_execute_member<T>::value;

    template <typename T>
    constexpr bool has_bulk_async_execute_member_v =
        has_bulk_async_execute_member<T>::value;

    template <typename T>
    constexpr bool has_bulk_then_execute_member_v =
        has_bulk_then_execute_member<T>::value;
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_context
    {
        using type = typename std::decay<decltype(
            std::declval<Executor const&>().context())>::type;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Components which create groups of execution agents may use execution
    // categories to communicate the forward progress and ordering guarantees
    // of these execution agents with respect to other agents within the same
    // group.

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_execution_category
    {
    private:
        template <typename T>
        using execution_category = typename T::execution_category;

    public:
        using type = hpx::util::detected_or_t<unsequenced_execution_tag,
            execution_category, Executor>;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_shape
    {
    private:
        template <typename T>
        using shape_type = typename T::shape_type;

    public:
        using type =
            hpx::util::detected_or_t<std::size_t, shape_type, Executor>;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_index
    {
    private:
        // exposition only
        template <typename T>
        using index_type = typename T::index_type;

    public:
        using type =
            hpx::util::detected_or_t<typename executor_shape<Executor>::type,
                index_type, Executor>;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_parameters_type
    {
    private:
        template <typename T>
        using parameters_type = typename T::parameters_type;

    public:
        using type =
            hpx::util::detected_or_t<parallel::execution::static_chunk_size,
                parameters_type, Executor>;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Executor, typename T, typename Ts,
            typename Enable = void>
        struct executor_future;

        template <typename Executor, typename T, typename Enable = void>
        struct exposes_future_type : std::false_type
        {
        };

        template <typename Executor, typename T>
        struct exposes_future_type<Executor, T,
            typename hpx::util::always_void<
                typename Executor::template future_type<T>>::type>
          : std::true_type
        {
        };

        template <typename Executor, typename T, typename... Ts>
        struct executor_future<Executor, T, hpx::util::pack<Ts...>,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value &&
                exposes_future_type<Executor, T>::value>::type>
        {
            using type = typename Executor::template future_type<T>;
        };

        template <typename Executor, typename T, typename... Ts>
        struct executor_future<Executor, T, hpx::util::pack<Ts...>,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value &&
                !exposes_future_type<Executor, T>::value>::type>
        {
            using type = decltype(std::declval<Executor&&>().async_execute(
                std::declval<T (*)(Ts...)>(), std::declval<Ts>()...));
        };

        template <typename Executor, typename T, typename Ts>
        struct executor_future<Executor, T, Ts,
            typename std::enable_if<
                !hpx::traits::is_two_way_executor<Executor>::value>::type>
        {
            using type = hpx::lcos::future<T>;
        };
    }    // namespace detail

    template <typename Executor, typename T, typename... Ts>
    struct executor_future
      : detail::executor_future<typename std::decay<Executor>::type, T,
            hpx::util::pack<typename std::decay<Ts>::type...>>
    {
    };

    template <typename Executor, typename T, typename... Ts>
    using executor_future_t =
        typename executor_future<Executor, T, Ts...>::type;

}}}    // namespace hpx::parallel::execution

namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct has_post_member
      : parallel::execution::has_post_member<typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_sync_execute_member
      : parallel::execution::has_sync_execute_member<
            typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_async_execute_member
      : parallel::execution::has_async_execute_member<
            typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_then_execute_member
      : parallel::execution::has_then_execute_member<
            typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_sync_execute_member
      : parallel::execution::has_bulk_sync_execute_member<
            typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_async_execute_member
      : parallel::execution::has_bulk_async_execute_member<
            typename std::decay<T>::type>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_then_execute_member
      : parallel::execution::has_bulk_then_execute_member<
            typename std::decay<T>::type>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename Enable = void>
    struct executor_context
      : parallel::execution::executor_context<
            typename std::decay<Executor>::type>
    {
    };

    template <typename Executor>
    using executor_context_t = typename executor_context<Executor>::type;

    template <typename Executor, typename Enable = void>
    struct executor_execution_category
      : parallel::execution::executor_execution_category<
            typename std::decay<Executor>::type>
    {
    };

    template <typename Executor>
    using executor_execution_category_t =
        typename executor_execution_category<Executor>::type;

    template <typename Executor, typename Enable = void>
    struct executor_shape
      : parallel::execution::executor_shape<typename std::decay<Executor>::type>
    {
    };

    template <typename Executor>
    using executor_shape_t = typename executor_shape<Executor>::type;

    template <typename Executor, typename Enable = void>
    struct executor_index
      : parallel::execution::executor_index<typename std::decay<Executor>::type>
    {
    };

    template <typename Executor>
    using executor_index_t = typename executor_index<Executor>::type;

    template <typename Executor, typename T, typename... Ts>
    struct executor_future
      : parallel::execution::executor_future<
            typename std::decay<Executor>::type, T,
            typename std::decay<Ts>::type...>
    {
    };

    template <typename Executor, typename T, typename... Ts>
    using executor_future_t =
        typename executor_future<Executor, T, Ts...>::type;

    ///////////////////////////////////////////////////////////////////////////
    // extension
    template <typename Executor, typename Enable = void>
    struct executor_parameters_type
      : parallel::execution::executor_parameters_type<
            typename std::decay<Executor>::type>
    {
    };

    template <typename Executor>
    using executor_parameters_type_t =
        typename executor_parameters_type<Executor>::type;
}}    // namespace hpx::traits

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
namespace hpx { namespace threads {
    class executor;
}}    // namespace hpx::threads

namespace hpx { namespace traits {
    namespace detail {
        template <typename Executor>
        struct is_threads_executor
          : std::is_base_of<threads::executor, Executor>
        {
        };
    }    // namespace detail

    template <typename Executor>
    struct is_threads_executor
      : detail::is_threads_executor<typename hpx::util::decay<Executor>::type>
    {
    };

    template <typename Policy>
    struct is_launch_policy_or_executor
      : std::integral_constant<bool,
            is_launch_policy<Policy>::value ||
                is_threads_executor<Policy>::value>
    {
    };

    //////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_execution_category<Executor,
        typename std::enable_if<is_threads_executor<Executor>::value>::type>
    {
        using type = parallel::execution::parallel_execution_tag;
    };
}}    // namespace hpx::traits
#endif
