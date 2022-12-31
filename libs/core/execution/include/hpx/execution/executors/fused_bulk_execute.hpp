//  Copyright (c) 2017-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::parallel::execution::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Shape, typename Future, typename... Ts>
    struct then_bulk_function_result
    {
        using value_type =
            typename hpx::traits::range_traits<Shape>::value_type;
        using type = hpx::util::detail::invoke_deferred_result_t<F, value_type,
            Future, Ts...>;
    };

    template <typename F, typename Shape, typename Future, typename... Ts>
    using then_bulk_function_result_t =
        typename then_bulk_function_result<F, Shape, Future, Ts...>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Shape, typename Future, bool IsVoid,
        typename... Ts>
    struct bulk_then_execute_result_impl;

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result_impl<F, Shape, Future, false, Ts...>
    {
        using type =
            std::vector<then_bulk_function_result_t<F, Shape, Future, Ts...>>;
    };

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result_impl<F, Shape, Future, true, Ts...>
    {
        using type = void;
    };

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result
      : bulk_then_execute_result_impl<F, Shape, Future,
            std::is_void_v<
                then_bulk_function_result_t<F, Shape, Future, Ts...>>,
            Ts...>
    {
    };

    template <typename F, typename Shape, typename Future, typename... Ts>
    using bulk_then_execute_result_t =
        typename bulk_then_execute_result<F, Shape, Future, Ts...>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename F, typename Shape, typename Future,
        std::size_t... Is, typename... Ts>
    HPX_FORCEINLINE auto fused_bulk_sync_execute(Executor&& exec, F&& f,
        Shape const& shape, Future&& predecessor, hpx::util::index_pack<Is...>,
        hpx::tuple<Ts...> const& args)
        -> decltype(execution::bulk_sync_execute(HPX_FORWARD(Executor, exec),
            HPX_FORWARD(F, f), shape, HPX_FORWARD(Future, predecessor),
            hpx::get<Is>(args)...))
    {
        return execution::bulk_sync_execute(HPX_FORWARD(Executor, exec),
            HPX_FORWARD(F, f), shape, HPX_FORWARD(Future, predecessor),
            hpx::get<Is>(args)...);
    }

    template <typename Executor, typename F, typename Shape, typename Args>
    struct fused_bulk_sync_execute_helper;

    template <typename Executor, typename F, typename Shape, typename... Ts>
    struct fused_bulk_sync_execute_helper<Executor, F, Shape, hpx::tuple<Ts...>>
    {
        Executor exec_;
        F f_;
        Shape shape_;
        hpx::tuple<Ts...> args_;

        template <typename Future>
        decltype(auto) operator()(Future&& predecessor)
        {
            return fused_bulk_sync_execute(exec_, f_, shape_,
                HPX_FORWARD(Future, predecessor),
                hpx::util::make_index_pack_t<sizeof...(Ts)>(), args_);
        }
    };

    template <typename Executor, typename F, typename Shape, typename Args>
    fused_bulk_sync_execute_helper<std::decay_t<Executor>, std::decay_t<F>,
        Shape, std::decay_t<Args>>
    make_fused_bulk_sync_execute_helper(
        Executor&& exec, F&& f, Shape const& shape, Args&& args)
    {
        return fused_bulk_sync_execute_helper<std::decay_t<Executor>,
            std::decay_t<F>, Shape, std::decay_t<Args>>{
            HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f), shape,
            HPX_FORWARD(Args, args)};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename F, typename Shape, typename Future,
        std::size_t... Is, typename... Ts>
    HPX_FORCEINLINE auto fused_bulk_async_execute(Executor&& exec, F&& f,
        Shape const& shape, Future&& predecessor, hpx::util::index_pack<Is...>,
        hpx::tuple<Ts...> const& args)
        -> decltype(execution::bulk_async_execute(HPX_FORWARD(Executor, exec),
            HPX_FORWARD(F, f), shape, HPX_FORWARD(Future, predecessor),
            hpx::get<Is>(args)...))
    {
        return execution::bulk_async_execute(HPX_FORWARD(Executor, exec),
            HPX_FORWARD(F, f), shape, HPX_FORWARD(Future, predecessor),
            hpx::get<Is>(args)...);
    }

    template <typename Executor, typename F, typename Shape, typename Args>
    struct fused_bulk_async_execute_helper;

    template <typename Executor, typename F, typename Shape, typename... Ts>
    struct fused_bulk_async_execute_helper<Executor, F, Shape,
        hpx::tuple<Ts...>>
    {
        Executor exec_;
        F f_;
        Shape shape_;
        hpx::tuple<Ts...> args_;

        template <typename Future>
        decltype(auto) operator()(Future&& predecessor)
        {
            return fused_bulk_async_execute(exec_, f_, shape_,
                HPX_FORWARD(Future, predecessor),
                hpx::util::make_index_pack_t<sizeof...(Ts)>(), args_);
        }
    };

    template <typename Executor, typename F, typename Shape, typename Args>
    fused_bulk_async_execute_helper<std::decay_t<Executor>, std::decay_t<F>,
        std::decay_t<Shape>, std::decay_t<Args>>
    make_fused_bulk_async_execute_helper(
        Executor&& exec, F&& f, Shape&& shape, Args&& args)
    {
        return fused_bulk_async_execute_helper<std::decay_t<Executor>,
            std::decay_t<F>, std::decay_t<Shape>, std::decay_t<Args>>{
            HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f),
            HPX_FORWARD(Shape, shape), HPX_FORWARD(Args, args)};
    }
}    // namespace hpx::parallel::execution::detail
