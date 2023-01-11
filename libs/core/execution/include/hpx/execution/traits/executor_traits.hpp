//  Copyright (c) 2017-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/type_support/detected.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx {

    template <typename R>
    class future;
}    // namespace hpx

namespace hpx::execution {

    namespace experimental {

        struct adaptive_static_chunk_size;
        struct auto_chunk_size;
        struct dynamic_chunk_size;
        struct guided_chunk_size;
        struct persistent_auto_chunk_size;
        struct static_chunk_size;
        struct num_cores;
    }    // namespace experimental

    ///////////////////////////////////////////////////////////////////////////
    struct sequenced_execution_tag;
    struct parallel_execution_tag;
    struct unsequenced_execution_tag;
}    // namespace hpx::execution

namespace hpx::parallel::execution {

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
    struct has_post_member : detail::has_post<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_sync_execute_member : detail::has_sync_execute<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_async_execute_member : detail::has_async_execute<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_then_execute_member : detail::has_then_execute<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_sync_execute_member
      : detail::has_bulk_sync_execute<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_async_execute_member
      : detail::has_bulk_async_execute<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_then_execute_member
      : detail::has_bulk_then_execute<std::decay_t<T>>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_context
    {
        using type =
            std::decay_t<decltype(std::declval<Executor const&>().context())>;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Components that create groups of execution agents may use execution
    // categories to communicate the forward progress and ordering guarantees of
    // these execution agents with respect to other agents within the same
    // group.

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_execution_category
    {
    private:
        template <typename T>
        using execution_category = typename T::execution_category;

    public:
        using type =
            hpx::util::detected_or_t<hpx::execution::unsequenced_execution_tag,
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
        using type = hpx::util::detected_or_t<
            hpx::execution::experimental::static_chunk_size, parameters_type,
            Executor>;
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
            std::void_t<typename Executor::template future_type<T>>>
          : std::true_type
        {
        };

        template <typename Executor, typename T, typename... Ts>
        struct executor_future<Executor, T, hpx::util::pack<Ts...>,
            std::enable_if_t<hpx::traits::is_two_way_executor_v<Executor> &&
                exposes_future_type<Executor, T>::value>>
        {
            using type = typename Executor::template future_type<T>;
        };

        template <typename Executor, typename T, typename... Ts>
        struct executor_future<Executor, T, hpx::util::pack<Ts...>,
            std::enable_if_t<hpx::traits::is_two_way_executor_v<Executor> &&
                has_async_execute_member<Executor>::value &&
                !exposes_future_type<Executor, T>::value>>
        {
            using type = decltype(std::declval<Executor&&>().async_execute(
                std::declval<T (*)(Ts...)>(), std::declval<Ts>()...));
        };

        template <typename Executor, typename T, typename... Ts>
        struct executor_future<Executor, T, hpx::util::pack<Ts...>,
            std::enable_if_t<hpx::traits::is_two_way_executor_v<Executor> &&
                !has_async_execute_member<Executor>::value &&
                !exposes_future_type<Executor, T>::value>>
        {
            using type = hpx::functional::tag_invoke_result_t<
                hpx::parallel::execution::async_execute_t, Executor,
                T (*)(Ts...), Ts...>;
        };

        template <typename Executor, typename T, typename Ts>
        struct executor_future<Executor, T, Ts,
            std::enable_if_t<!hpx::traits::is_two_way_executor_v<Executor>>>
        {
            using type = hpx::future<T>;
        };
    }    // namespace detail

    template <typename Executor, typename T, typename... Ts>
    struct executor_future
      : detail::executor_future<std::decay_t<Executor>, T,
            hpx::util::pack<std::decay_t<Ts>...>>
    {
    };

    template <typename Executor, typename T, typename... Ts>
    using executor_future_t =
        typename executor_future<Executor, T, Ts...>::type;
}    // namespace hpx::parallel::execution

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct has_post_member
      : parallel::execution::has_post_member<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_sync_execute_member
      : parallel::execution::has_sync_execute_member<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_async_execute_member
      : parallel::execution::has_async_execute_member<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_then_execute_member
      : parallel::execution::has_then_execute_member<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_sync_execute_member
      : parallel::execution::has_bulk_sync_execute_member<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_async_execute_member
      : parallel::execution::has_bulk_async_execute_member<std::decay_t<T>>
    {
    };

    template <typename T, typename Enable = void>
    struct has_bulk_then_execute_member
      : parallel::execution::has_bulk_then_execute_member<std::decay_t<T>>
    {
    };

    template <typename T>
    inline constexpr bool has_post_member_v = has_post_member<T>::value;

    template <typename T>
    inline constexpr bool has_sync_execute_member_v =
        has_sync_execute_member<T>::value;

    template <typename T>
    inline constexpr bool has_async_execute_member_v =
        has_async_execute_member<T>::value;

    template <typename T>
    inline constexpr bool has_then_execute_member_v =
        has_then_execute_member<T>::value;

    template <typename T>
    inline constexpr bool has_bulk_sync_execute_member_v =
        has_bulk_sync_execute_member<T>::value;

    template <typename T>
    inline constexpr bool has_bulk_async_execute_member_v =
        has_bulk_async_execute_member<T>::value;

    template <typename T>
    inline constexpr bool has_bulk_then_execute_member_v =
        has_bulk_then_execute_member<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename Enable = void>
    struct executor_context
      : parallel::execution::executor_context<std::decay_t<Executor>>
    {
    };

    template <typename Executor>
    using executor_context_t = typename executor_context<Executor>::type;

    template <typename Executor, typename Enable = void>
    struct executor_execution_category
      : parallel::execution::executor_execution_category<std::decay_t<Executor>>
    {
    };

    template <typename Executor>
    using executor_execution_category_t =
        typename executor_execution_category<Executor>::type;

    template <typename Executor, typename Enable = void>
    struct executor_shape
      : parallel::execution::executor_shape<std::decay_t<Executor>>
    {
    };

    template <typename Executor>
    using executor_shape_t = typename executor_shape<Executor>::type;

    template <typename Executor, typename Enable = void>
    struct executor_index
      : parallel::execution::executor_index<std::decay_t<Executor>>
    {
    };

    template <typename Executor>
    using executor_index_t = typename executor_index<Executor>::type;

    template <typename Executor, typename T, typename... Ts>
    struct executor_future
      : parallel::execution::executor_future<std::decay_t<Executor>, T,
            std::decay_t<Ts>...>
    {
    };

    template <typename Executor, typename T, typename... Ts>
    using executor_future_t =
        typename executor_future<Executor, T, Ts...>::type;

    ///////////////////////////////////////////////////////////////////////////
    // extension
    template <typename Executor, typename Enable = void>
    struct executor_parameters_type
      : parallel::execution::executor_parameters_type<std::decay_t<Executor>>
    {
    };

    template <typename Executor>
    using executor_parameters_type_t =
        typename executor_parameters_type<Executor>::type;
}    // namespace hpx::traits
