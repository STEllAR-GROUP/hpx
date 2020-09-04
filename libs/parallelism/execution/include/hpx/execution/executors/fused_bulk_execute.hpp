//  Copyright (c) 2017-2019 Hartmut Kaiser
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

namespace hpx { namespace parallel { namespace execution { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Shape, typename Future, typename... Ts>
    struct then_bulk_function_result
    {
        typedef
            typename hpx::traits::range_traits<Shape>::value_type value_type;
        typedef typename hpx::util::detail::invoke_deferred_result<F,
            value_type, Future, Ts...>::type type;
    };

    template <typename F, typename Shape, typename Future, bool IsVoid,
        typename... Ts>
    struct bulk_then_execute_result_impl;

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result_impl<F, Shape, Future, false, Ts...>
    {
        typedef std::vector<
            typename then_bulk_function_result<F, Shape, Future, Ts...>::type>
            type;
    };

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result_impl<F, Shape, Future, true, Ts...>
    {
        typedef void type;
    };

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result
      : bulk_then_execute_result_impl<F, Shape, Future,
            std::is_void<typename then_bulk_function_result<F, Shape, Future,
                Ts...>::type>::value,
            Ts...>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename F, typename Shape, typename Future,
        std::size_t... Is, typename... Ts>
    HPX_FORCEINLINE auto fused_bulk_sync_execute(Executor&& exec, F&& f,
        Shape const& shape, Future&& predecessor, hpx::util::index_pack<Is...>,
        hpx::tuple<Ts...> const& args)
        -> decltype(execution::bulk_sync_execute(std::forward<Executor>(exec),
            std::forward<F>(f), shape, std::forward<Future>(predecessor),
            hpx::get<Is>(args)...))
    {
        return execution::bulk_sync_execute(std::forward<Executor>(exec),
            std::forward<F>(f), shape, std::forward<Future>(predecessor),
            hpx::get<Is>(args)...);
    }

    template <typename Result, typename Executor, typename F, typename Shape,
        typename Args>
    struct fused_bulk_sync_execute_helper;

    template <typename Result, typename Executor, typename F, typename Shape,
        typename... Ts>
    struct fused_bulk_sync_execute_helper<Result, Executor, F, Shape,
        hpx::tuple<Ts...>>
    {
        Executor exec_;
        F f_;
        Shape shape_;
        hpx::tuple<Ts...> args_;

        template <typename Future>
        Result operator()(Future&& predecessor)
        {
            return fused_bulk_sync_execute(exec_, f_, shape_,
                std::forward<Future>(predecessor),
                typename hpx::util::make_index_pack<sizeof...(Ts)>::type(),
                args_);
        }
    };

    template <typename Result, typename Executor, typename F, typename Shape,
        typename Args>
    fused_bulk_sync_execute_helper<Result, typename std::decay<Executor>::type,
        typename std::decay<F>::type, Shape, typename std::decay<Args>::type>
    make_fused_bulk_sync_execute_helper(
        Executor&& exec, F&& f, Shape const& shape, Args&& args)
    {
        return fused_bulk_sync_execute_helper<Result,
            typename std::decay<Executor>::type, typename std::decay<F>::type,
            Shape, typename std::decay<Args>::type>{
            std::forward<Executor>(exec), std::forward<F>(f), shape,
            std::forward<Args>(args)};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename F, typename Shape, typename Future,
        std::size_t... Is, typename... Ts>
    HPX_FORCEINLINE auto fused_bulk_async_execute(Executor&& exec, F&& f,
        Shape const& shape, Future&& predecessor, hpx::util::index_pack<Is...>,
        hpx::tuple<Ts...> const& args)
        -> decltype(execution::bulk_async_execute(std::forward<Executor>(exec),
            std::forward<F>(f), shape, std::forward<Future>(predecessor),
            hpx::get<Is>(args)...))
    {
        return execution::bulk_async_execute(std::forward<Executor>(exec),
            std::forward<F>(f), shape, std::forward<Future>(predecessor),
            hpx::get<Is>(args)...);
    }

    template <typename Result, typename Executor, typename F, typename Shape,
        typename Args>
    struct fused_bulk_async_execute_helper;

    template <typename Result, typename Executor, typename F, typename Shape,
        typename... Ts>
    struct fused_bulk_async_execute_helper<Result, Executor, F, Shape,
        hpx::tuple<Ts...>>
    {
        Executor exec_;
        F f_;
        Shape shape_;
        hpx::tuple<Ts...> args_;

        template <typename Future>
        Result operator()(Future&& predecessor)
        {
            return fused_bulk_async_execute(exec_, f_, shape_,
                std::forward<Future>(predecessor),
                typename hpx::util::make_index_pack<sizeof...(Ts)>::type(),
                args_);
        }
    };

    template <typename Result, typename Executor, typename F, typename Shape,
        typename Args>
    fused_bulk_async_execute_helper<Result, typename std::decay<Executor>::type,
        typename std::decay<F>::type, typename std::decay<Shape>::type,
        typename std::decay<Args>::type>
    make_fused_bulk_async_execute_helper(
        Executor&& exec, F&& f, Shape&& shape, Args&& args)
    {
        return fused_bulk_async_execute_helper<Result,
            typename std::decay<Executor>::type, typename std::decay<F>::type,
            typename std::decay<Shape>::type, typename std::decay<Args>::type>{
            std::forward<Executor>(exec), std::forward<F>(f),
            std::forward<Shape>(shape), std::forward<Args>(args)};
    }
}}}}    // namespace hpx::parallel::execution::detail
