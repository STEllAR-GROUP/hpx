//  Copyright (c) 2017 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/execution.hpp

#pragma once

#include <hpx/config.hpp>
// Necessary to avoid circular include
#include <hpx/execution_base/execution.hpp>

#include <hpx/assert.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/executors/fused_bulk_execute.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution/traits/future_then_result_exec.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL

    // customization point for OneWayExecutor interface
    // execute()
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename... Ts>
        struct sync_execute_not_callable;

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the sync_execute() customization point
        template <typename Executor, typename F, typename... Ts>
        HPX_FORCEINLINE auto sync_execute_dispatch(
            hpx::traits::detail::wrap_int, Executor&& /* exec */, F&& /* f */,
            Ts&&... /* ts */) -> sync_execute_not_callable<Executor, F, Ts...>
        {
            return sync_execute_not_callable<Executor, F, Ts...>{};
        }

        template <typename OneWayExecutor, typename F, typename... Ts>
        HPX_FORCEINLINE auto sync_execute_dispatch(int, OneWayExecutor&& exec,
            F&& f, Ts&&... ts) -> decltype(exec.sync_execute(std::forward<F>(f),
            std::forward<Ts>(ts)...))
        {
            return exec.sync_execute(
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        // emulate async_execute() on OneWayExecutors
        template <typename Executor>
        struct async_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
                !hpx::traits::is_two_way_executor<Executor>::value>::type>
        {
            template <typename OneWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call_impl(
                std::false_type, OneWayExecutor&& exec, F&& f, Ts&&... ts)
                -> hpx::future<decltype(
                    sync_execute_dispatch(0, std::forward<OneWayExecutor>(exec),
                        std::forward<F>(f), std::forward<Ts>(ts)...))>
            {
                return hpx::lcos::make_ready_future(
                    sync_execute_dispatch(0, std::forward<OneWayExecutor>(exec),
                        std::forward<F>(f), std::forward<Ts>(ts)...));
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static hpx::future<void> call_impl(
                std::true_type, OneWayExecutor&& exec, F&& f, Ts&&... ts)
            {
                sync_execute_dispatch(0, std::forward<OneWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
                return hpx::lcos::make_ready_future();
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call(OneWayExecutor&& exec, F&& f,
                Ts&&... ts) -> hpx::future<decltype(sync_execute_dispatch(0,
                std::forward<OneWayExecutor>(exec), std::forward<F>(f),
                std::forward<Ts>(ts)...))>
            {
                typedef std::is_void<decltype(
                    sync_execute_dispatch(0, std::forward<OneWayExecutor>(exec),
                        std::forward<F>(f), std::forward<Ts>(ts)...))>
                    is_void;

                return call_impl(is_void(), exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<OneWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };

        // emulate sync_execute() on OneWayExecutors
        template <typename Executor>
        struct sync_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
                !hpx::traits::is_two_way_executor<Executor>::value>::type>
        {
            template <typename OneWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call(OneWayExecutor&& exec, F&& f,
                Ts&&... ts) -> decltype(sync_execute_dispatch(0,
                std::forward<OneWayExecutor>(exec), std::forward<F>(f),
                std::forward<Ts>(ts)...))
            {
                return sync_execute_dispatch(0,
                    std::forward<OneWayExecutor>(exec), std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<OneWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };

        // emulate then_execute() on OneWayExecutors
        template <typename Executor>
        struct then_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
                !hpx::traits::is_two_way_executor<Executor>::value>::type>
        {
            template <typename OneWayExecutor, typename F, typename Future,
                typename... Ts>
            HPX_FORCEINLINE static hpx::lcos::future<typename hpx::util::
                    detail::invoke_deferred_result<F, Future, Ts...>::type>
            call(OneWayExecutor&& exec, F&& f, Future&& predecessor, Ts&&... ts)
            {
                typedef typename hpx::util::detail::invoke_deferred_result<F,
                    Future, Ts...>::type result_type;

                auto func = hpx::util::one_shot(hpx::util::bind_back(
                    std::forward<F>(f), std::forward<Ts>(ts)...));

                typename hpx::traits::detail::shared_state_ptr<
                    result_type>::type p =
                    lcos::detail::make_continuation_exec<result_type>(
                        std::forward<Future>(predecessor),
                        std::forward<OneWayExecutor>(exec), std::move(func));

                return hpx::traits::future_access<
                    hpx::lcos::future<result_type>>::create(std::move(p));
            }

            template <typename OneWayExecutor, typename F, typename Future,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<OneWayExecutor>(), std::declval<F>(),
                        std::declval<Future>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        // emulate post() on OneWayExecutors
        template <typename Executor>
        struct post_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
                !hpx::traits::is_two_way_executor<Executor>::value &&
                !hpx::traits::is_never_blocking_one_way_executor<
                    Executor>::value>::type>
        {
            template <typename OneWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static void call_impl(hpx::traits::detail::wrap_int,
                OneWayExecutor&& exec, F&& f, Ts&&... ts)
            {
                // execute synchronously
                sync_execute_dispatch(0, std::forward<OneWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            // dispatch to V1 executors
            template <typename OneWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call_impl(int, OneWayExecutor&& exec,
                F&& f, Ts&&... ts) -> decltype(exec.post(std::forward<F>(f),
                std::forward<Ts>(ts)...))
            {
                // use post, if exposed
                return exec.post(std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call(OneWayExecutor&& exec, F&& f,
                Ts&&... ts) -> decltype(call_impl(0, exec, std::forward<F>(f),
                std::forward<Ts>(ts)...))
            {
                // simply discard the returned future
                return call_impl(0, std::forward<OneWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename OneWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<OneWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // customization points for TwoWayExecutor interface
    // async_execute(), sync_execute(), then_execute()
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename... Ts>
        struct async_execute_not_callable;

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the async_execute() customization point
        template <typename Executor, typename F, typename... Ts>
        HPX_FORCEINLINE auto async_execute_dispatch(
            hpx::traits::detail::wrap_int, Executor&& /* exec */, F&& /* f */,
            Ts&&... /* ts */) -> async_execute_not_callable<Executor, F, Ts...>
        {
            return async_execute_not_callable<Executor, F, Ts...>{};
        }

        template <typename TwoWayExecutor, typename F, typename... Ts>
        HPX_FORCEINLINE auto async_execute_dispatch(
            int, TwoWayExecutor&& exec, F&& f, Ts&&... ts)
            -> decltype(
                exec.async_execute(std::forward<F>(f), std::forward<Ts>(ts)...))
        {
            return exec.async_execute(
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor>
        struct async_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value>::type>
        {
            template <typename TwoWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call(TwoWayExecutor&& exec, F&& f,
                Ts&&... ts) -> decltype(async_execute_dispatch(0,
                std::forward<TwoWayExecutor>(exec), std::forward<F>(f),
                std::forward<Ts>(ts)...))
            {
                return async_execute_dispatch(0,
                    std::forward<TwoWayExecutor>(exec), std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<TwoWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the sync_execute() customization point
        template <typename Executor>
        struct sync_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value>::type>
        {
            // fall-back: emulate sync_execute using async_execute
            template <typename TwoWayExecutor, typename F, typename... Ts>
            static auto call_impl(std::false_type, TwoWayExecutor&& exec, F&& f,
                Ts&&... ts) -> typename hpx::util::invoke_result<F, Ts...>::type
            {
                try
                {
                    typedef
                        typename hpx::util::detail::invoke_deferred_result<F,
                            Ts...>::type result_type;

                    // use async execution, wait for result, propagate exceptions
                    return async_execute_dispatch(0,
                        std::forward<TwoWayExecutor>(exec),
                        [&]() -> result_type {
                            return HPX_INVOKE(
                                std::forward<F>(f), std::forward<Ts>(ts)...);
                        })
                        .get();
                }
                catch (std::bad_alloc const& ba)
                {
                    throw ba;
                }
                catch (...)
                {
                    throw hpx::exception_list(std::current_exception());
                }
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static void call_impl(
                std::true_type, TwoWayExecutor&& exec, F&& f, Ts&&... ts)
            {
                async_execute_dispatch(0, std::forward<TwoWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...)
                    .get();
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call_impl(hpx::traits::detail::wrap_int,
                TwoWayExecutor&& exec, F&& f, Ts&&... ts) ->
                typename hpx::util::invoke_result<F, Ts...>::type
            {
                typedef typename std::is_void<typename hpx::util::detail::
                        invoke_deferred_result<F, Ts...>::type>::type is_void;

                return call_impl(is_void(), std::forward<TwoWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call_impl(
                int, TwoWayExecutor&& exec, F&& f, Ts&&... ts)
                -> decltype(exec.sync_execute(
                    std::forward<F>(f), std::forward<Ts>(ts)...))
            {
                return exec.sync_execute(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call(
                TwoWayExecutor&& exec, F&& f, Ts&&... ts)
                -> decltype(call_impl(0, std::forward<TwoWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...))
            {
                return call_impl(0, std::forward<TwoWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<TwoWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        // then_execute()

        template <typename Executor>
        struct then_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value>::type>
        {
            template <typename TwoWayExecutor, typename F, typename Future,
                typename... Ts>
            static hpx::lcos::future<typename hpx::util::detail::
                    invoke_deferred_result<F, Future, Ts...>::type>
            call_impl(hpx::traits::detail::wrap_int, TwoWayExecutor&& exec,
                F&& f, Future&& predecessor, Ts&&... ts)
            {
                typedef typename hpx::util::detail::invoke_deferred_result<F,
                    Future, Ts...>::type result_type;

                auto func = hpx::util::one_shot(hpx::util::bind_back(
                    std::forward<F>(f), std::forward<Ts>(ts)...));

                typename hpx::traits::detail::shared_state_ptr<
                    result_type>::type p =
                    lcos::detail::make_continuation_exec<result_type>(
                        std::forward<Future>(predecessor),
                        std::forward<TwoWayExecutor>(exec), std::move(func));

                return hpx::traits::future_access<
                    hpx::lcos::future<result_type>>::create(std::move(p));
            }

            template <typename TwoWayExecutor, typename F, typename Future,
                typename... Ts>
            HPX_FORCEINLINE static auto call_impl(int, TwoWayExecutor&& exec,
                F&& f, Future&& predecessor, Ts&&... ts)
                -> decltype(exec.then_execute(std::forward<F>(f),
                    std::forward<Future>(predecessor), std::forward<Ts>(ts)...))
            {
                return exec.then_execute(std::forward<F>(f),
                    std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename Future,
                typename... Ts>
            HPX_FORCEINLINE static auto call(TwoWayExecutor&& exec, F&& f,
                Future&& predecessor, Ts&&... ts) -> decltype(call_impl(0,
                std::forward<TwoWayExecutor>(exec), std::forward<F>(f),
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...))
            {
                return call_impl(0, std::forward<TwoWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Future>(predecessor),
                    std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename Future,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<TwoWayExecutor>(), std::declval<F>(),
                        std::declval<Future>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        // emulate post() on TwoWayExecutors
        template <typename Executor>
        struct post_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value &&
                !hpx::traits::is_never_blocking_one_way_executor<
                    Executor>::value>::type>
        {
            template <typename TwoWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static void call_impl(hpx::traits::detail::wrap_int,
                TwoWayExecutor&& exec, F&& f, Ts&&... ts)
            {
                // simply discard the returned future
                exec.async_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            // dispatch to V1 executors
            template <typename TwoWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call_impl(int, TwoWayExecutor&& exec,
                F&& f, Ts&&... ts) -> decltype(exec.post(std::forward<F>(f),
                std::forward<Ts>(ts)...))
            {
                // use post, if exposed
                return exec.post(std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            HPX_FORCEINLINE static auto call(
                TwoWayExecutor&& exec, F&& f, Ts&&... ts)
                -> decltype(call_impl(0, std::forward<TwoWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...))
            {
                return call_impl(0, std::forward<TwoWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<TwoWayExecutor>(),
                    std::declval<F>(), std::declval<Ts>()...));
            };
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // post()
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // default implementation of the post() customization point

        template <typename Executor, typename F, typename... Ts>
        struct post_not_callable;

        template <typename Executor, typename F, typename... Ts>
        HPX_FORCEINLINE auto post_dispatch(hpx::traits::detail::wrap_int,
            Executor&& /* exec */, F&& /* f */, Ts&&... /* ts */)
            -> post_not_callable<Executor, F, Ts...>
        {
            return post_not_callable<Executor, F, Ts...>{};
        }

        // default implementation of the post() customization point
        template <typename NonBlockingOneWayExecutor, typename F,
            typename... Ts>
        HPX_FORCEINLINE auto post_dispatch(
            int, NonBlockingOneWayExecutor&& exec, F&& f, Ts&&... ts)
            -> decltype(exec.post(std::forward<F>(f), std::forward<Ts>(ts)...))
        {
            return exec.post(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor>
        struct post_fn_helper<Executor,
            typename std::enable_if<hpx::traits::
                    is_never_blocking_one_way_executor<Executor>::value>::type>
        {
            template <typename NonBlockingOneWayExecutor, typename F,
                typename... Ts>
            HPX_FORCEINLINE static auto call(NonBlockingOneWayExecutor&& exec,
                F&& f, Ts&&... ts) -> decltype(post_dispatch(0,
                std::forward<NonBlockingOneWayExecutor>(exec),
                std::forward<F>(f), std::forward<Ts>(ts)...))
            {
                return post_dispatch(0,
                    std::forward<NonBlockingOneWayExecutor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename NonBlockingOneWayExecutor, typename F,
                typename... Ts>
            struct result
            {
                using type =
                    decltype(call(std::declval<NonBlockingOneWayExecutor>(),
                        std::declval<F>(), std::declval<Ts>()...));
            };
        };
    }    // namespace detail
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // customization points for BulkTwoWayExecutor interface
    // bulk_async_execute(), bulk_sync_execute(), bulk_then_execute()

    /// \cond NOINTERNAL
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // bulk_async_execute()

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the bulk_async_execute() customization point

        template <typename Executor, typename F, typename Shape, typename... Ts>
        struct bulk_async_execute_not_callable;

        template <typename Executor, typename F, typename Shape, typename... Ts>
        auto bulk_async_execute_dispatch(hpx::traits::detail::wrap_int,
            Executor&& /* exec */, F&& /* f */, Shape const& /* shape */,
            Ts&&... /* ts */)
            -> bulk_async_execute_not_callable<Executor, F, Shape, Ts...>
        {
            return bulk_async_execute_not_callable<Executor, F, Shape, Ts...>{};
        }

        template <typename BulkTwoWayExecutor, typename F, typename Shape,
            typename... Ts>
        HPX_FORCEINLINE auto bulk_async_execute_dispatch(int,
            BulkTwoWayExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
            -> decltype(exec.bulk_async_execute(
                std::forward<F>(f), shape, std::forward<Ts>(ts)...))
        {
            return exec.bulk_async_execute(
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }

        template <typename F, typename Shape, typename... Ts>
        struct bulk_function_result
        {
            typedef typename hpx::traits::range_traits<Shape>::value_type
                value_type;
            typedef typename hpx::util::detail::invoke_deferred_result<F,
                value_type, Ts...>::type type;
        };

        template <typename Executor>
        struct bulk_async_execute_fn_helper<Executor,
            typename std::enable_if<
                (hpx::traits::is_one_way_executor<Executor>::value ||
                    hpx::traits::is_two_way_executor<Executor>::value) &&
                !hpx::traits::is_bulk_two_way_executor<Executor>::value>::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static auto call_impl(hpx::traits::detail::wrap_int,
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> std::vector<typename hpx::traits::executor_future<Executor,
                    typename bulk_function_result<F, Shape, Ts...>::type,
                    Ts...>::type>
            {
                std::vector<typename hpx::traits::executor_future<Executor,
                    typename bulk_function_result<F, Shape, Ts...>::type,
                    Ts...>::type>
                    results;
                results.reserve(hpx::util::size(shape));

                for (auto const& elem : shape)
                {
                    results.push_back(
                        execution::async_execute(exec, f, elem, ts...));
                }

                return results;
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            HPX_FORCEINLINE static auto call_impl(
                int, BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(exec.bulk_async_execute(
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...))
            {
                return exec.bulk_async_execute(
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            HPX_FORCEINLINE static auto call(
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(call_impl(0, std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...))
            {
                return call_impl(0, std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<BulkExecutor>(), std::declval<F>(),
                        std::declval<Shape const&>(), std::declval<Ts>()...));
            };
        };

        template <typename Executor>
        struct bulk_async_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_bulk_two_way_executor<Executor>::value>::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            HPX_FORCEINLINE static auto call(
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(bulk_async_execute_dispatch(0,
                    std::forward<BulkExecutor>(exec), std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...))
            {
                return bulk_async_execute_dispatch(0,
                    std::forward<BulkExecutor>(exec), std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<BulkExecutor>(), std::declval<F>(),
                        std::declval<Shape const&>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        // bulk_sync_execute()

        // default implementation of the bulk_sync_execute() customization point
        template <typename Executor, typename F, typename Shape, typename... Ts>
        struct bulk_sync_execute_not_callable;

        template <typename Executor, typename F, typename Shape, typename... Ts>
        auto bulk_sync_execute_dispatch(hpx::traits::detail::wrap_int,
            Executor&& /* exec */, F&& /* f */, Shape const& /* shape */,
            Ts&&... /* ts */)
            -> bulk_sync_execute_not_callable<Executor, F, Shape, Ts...>
        {
            return bulk_sync_execute_not_callable<Executor, F, Shape, Ts...>{};
        }

        template <typename BulkTwoWayExecutor, typename F, typename Shape,
            typename... Ts>
        HPX_FORCEINLINE auto bulk_sync_execute_dispatch(int,
            BulkTwoWayExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
            -> decltype(exec.bulk_sync_execute(
                std::forward<F>(f), shape, std::forward<Ts>(ts)...))
        {
            return exec.bulk_sync_execute(
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, bool IsVoid, typename... Ts>
        struct bulk_execute_result_impl;

        template <typename F, typename Shape, typename... Ts>
        struct bulk_execute_result_impl<F, Shape, false, Ts...>
        {
            typedef std::vector<
                typename bulk_function_result<F, Shape, Ts...>::type>
                type;
        };

        template <typename F, typename Shape, typename... Ts>
        struct bulk_execute_result_impl<F, Shape, true, Ts...>
        {
            typedef void type;
        };

        template <typename F, typename Shape, typename... Ts>
        struct bulk_execute_result
          : bulk_execute_result_impl<F, Shape,
                std::is_void<typename bulk_function_result<F, Shape,
                    Ts...>::type>::value,
                Ts...>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        struct bulk_sync_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
                !hpx::traits::is_two_way_executor<Executor>::value &&
                !hpx::traits::is_bulk_one_way_executor<Executor>::value>::type>
        {
            // returns void if F returns void
            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static auto call_impl(std::false_type, BulkExecutor&& exec, F&& f,
                Shape const& shape, Ts&&... ts) ->
                typename bulk_execute_result_impl<F, Shape, false, Ts...>::type
            {
                try
                {
                    typename bulk_execute_result_impl<F, Shape, false,
                        Ts...>::type results;
                    results.reserve(hpx::util::size(shape));

                    for (auto const& elem : shape)
                    {
                        results.push_back(
                            execution::sync_execute(exec, f, elem, ts...));
                    }
                    return results;
                }
                catch (std::bad_alloc const& ba)
                {
                    throw ba;
                }
                catch (...)
                {
                    throw exception_list(std::current_exception());
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static void call_impl(std::true_type, BulkExecutor&& exec, F&& f,
                Shape const& shape, Ts&&... ts)
            {
                try
                {
                    for (auto const& elem : shape)
                    {
                        execution::sync_execute(exec, f, elem, ts...);
                    }
                }
                catch (std::bad_alloc const& ba)
                {
                    throw ba;
                }
                catch (...)
                {
                    throw exception_list(std::current_exception());
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            HPX_FORCEINLINE static auto call_impl(hpx::traits::detail::wrap_int,
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts) ->
                typename bulk_execute_result<F, Shape, Ts...>::type
            {
                typedef typename std::is_void<typename bulk_function_result<F,
                    Shape, Ts...>::type>::type is_void;

                return call_impl(is_void(), std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static auto call_impl(
                int, BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(exec.bulk_sync_execute(
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...))
            {
                return exec.bulk_sync_execute(
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            HPX_FORCEINLINE static auto call(
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(call_impl(0, std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...))
            {
                return call_impl(0, std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<BulkExecutor>(), std::declval<F>(),
                        std::declval<Shape const&>(), std::declval<Ts>()...));
            };
        };

        template <typename Executor>
        struct bulk_sync_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value &&
                !hpx::traits::is_bulk_one_way_executor<Executor>::value>::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static auto call_impl(std::false_type, BulkExecutor&& exec, F&& f,
                Shape const& shape, Ts&&... ts) ->
                typename bulk_execute_result<F, Shape, Ts...>::type
            {
                typedef std::vector<typename hpx::traits::executor_future<
                    Executor,
                    typename bulk_function_result<F, Shape, Ts...>::type>::type>
                    result_type;

                try
                {
                    result_type results;
                    results.reserve(hpx::util::size(shape));
                    for (auto const& elem : shape)
                    {
                        results.push_back(
                            execution::async_execute(exec, f, elem, ts...));
                    }
                    return hpx::util::unwrap(results);
                }
                catch (std::bad_alloc const& ba)
                {
                    throw ba;
                }
                catch (...)
                {
                    throw exception_list(std::current_exception());
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            static void call_impl(std::true_type, BulkExecutor&& exec, F&& f,
                Shape const& shape, Ts&&... ts)
            {
                typedef std::vector<typename hpx::traits::executor_future<
                    Executor,
                    typename bulk_function_result<F, Shape, Ts...>::type>::type>
                    result_type;

                try
                {
                    result_type results;
                    results.reserve(hpx::util::size(shape));

                    for (auto const& elem : shape)
                    {
                        results.push_back(
                            execution::async_execute(exec, f, elem, ts...));
                    }
                    hpx::lcos::wait_all(std::move(results));
                }
                catch (std::bad_alloc const& ba)
                {
                    throw ba;
                }
                catch (...)
                {
                    throw exception_list(std::current_exception());
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            HPX_FORCEINLINE static auto call_impl(hpx::traits::detail::wrap_int,
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts) ->
                typename bulk_execute_result<F, Shape, Ts...>::type
            {
                typedef typename std::is_void<typename bulk_function_result<F,
                    Shape, Ts...>::type>::type is_void;

                return call_impl(is_void(), std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            HPX_FORCEINLINE static auto call_impl(
                int, BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(exec.bulk_sync_execute(
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...))
            {
                return exec.bulk_sync_execute(
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            HPX_FORCEINLINE static auto call(
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(call_impl(0, std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...))
            {
                return call_impl(0, std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<BulkExecutor>(), std::declval<F>(),
                        std::declval<Shape const&>(), std::declval<Ts>()...));
            };
        };

        template <typename Executor>
        struct bulk_sync_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_bulk_one_way_executor<Executor>::value>::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            HPX_FORCEINLINE static auto call(
                BulkExecutor&& exec, F&& f, Shape const& shape, Ts&&... ts)
                -> decltype(bulk_sync_execute_dispatch(0,
                    std::forward<BulkExecutor>(exec), std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...))
            {
                return bulk_sync_execute_dispatch(0,
                    std::forward<BulkExecutor>(exec), std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename... Ts>
            struct result
            {
                using type = decltype(
                    call(std::declval<BulkExecutor>(), std::declval<F>(),
                        std::declval<Shape const&>(), std::declval<Ts>()...));
            };
        };
    }    // namespace detail
    /// \endcond

    /// \cond NOINTERNAL
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // bulk_then_execute()

        template <typename Executor>
        struct bulk_then_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
                !hpx::traits::is_bulk_two_way_executor<Executor>::value>::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            static auto call_impl(std::false_type, BulkExecutor&& exec, F&& f,
                Shape const& shape, Future&& predecessor, Ts&&... ts)
                -> hpx::future<typename bulk_then_execute_result<F, Shape,
                    Future, Ts...>::type>
            {
                typedef typename bulk_then_execute_result<F, Shape, Future,
                    Ts...>::type result_type;

                typedef typename hpx::traits::detail::shared_state_ptr<
                    result_type>::type shared_state_type;

                auto func = make_fused_bulk_sync_execute_helper<result_type>(
                    exec, std::forward<F>(f), shape,
                    hpx::make_tuple(std::forward<Ts>(ts)...));

                shared_state_type p =
                    lcos::detail::make_continuation_exec<result_type>(
                        std::forward<Future>(predecessor),
                        std::forward<BulkExecutor>(exec), std::move(func));

                return hpx::traits::future_access<
                    hpx::future<result_type>>::create(std::move(p));
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            static hpx::future<void> call_impl(std::true_type,
                BulkExecutor&& exec, F&& f, Shape const& shape,
                Future&& predecessor, Ts&&... ts)
            {
                auto func = make_fused_bulk_sync_execute_helper<void>(exec,
                    std::forward<F>(f), shape,
                    hpx::make_tuple(std::forward<Ts>(ts)...));

                typename hpx::traits::detail::shared_state_ptr<void>::type p =
                    lcos::detail::make_continuation_exec<void>(
                        std::forward<Future>(predecessor),
                        std::forward<BulkExecutor>(exec), std::move(func));

                return hpx::traits::future_access<hpx::future<void>>::create(
                    std::move(p));
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            HPX_FORCEINLINE static auto call_impl(hpx::traits::detail::wrap_int,
                BulkExecutor&& exec, F&& f, Shape const& shape,
                Future&& predecessor, Ts&&... ts)
                -> hpx::future<typename bulk_then_execute_result<F, Shape,
                    Future, Ts...>::type>
            {
                typedef
                    typename std::is_void<typename then_bulk_function_result<F,
                        Shape, Future, Ts...>::type>::type is_void;

                return bulk_then_execute_fn_helper::call_impl(is_void(),
                    std::forward<BulkExecutor>(exec), std::forward<F>(f), shape,
                    std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            HPX_FORCEINLINE static auto call_impl(int, BulkExecutor&& exec,
                F&& f, Shape const& shape, Future&& predecessor, Ts&&... ts)
                -> decltype(exec.bulk_then_execute(std::forward<F>(f), shape,
                    std::forward<Future>(predecessor), std::forward<Ts>(ts)...))
            {
                return exec.bulk_then_execute(std::forward<F>(f), shape,
                    std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            HPX_FORCEINLINE static auto call(BulkExecutor&& exec, F&& f,
                Shape const& shape, Future&& predecessor, Ts&&... ts)
                -> decltype(call_impl(0, std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape,
                    hpx::lcos::make_shared_future(
                        std::forward<Future>(predecessor)),
                    std::forward<Ts>(ts)...))
            {
                return call_impl(0, std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape,
                    hpx::lcos::make_shared_future(
                        std::forward<Future>(predecessor)),
                    std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<BulkExecutor>(),
                    std::declval<F>(), std::declval<Shape const&>(),
                    std::declval<Future>(), std::declval<Ts>()...));
            };
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        struct bulk_then_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_bulk_two_way_executor<Executor>::value>::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            static auto call_impl(
                hpx::traits::detail::wrap_int, BulkExecutor&& exec, F&& f,
                Shape const& shape, Future&& predecessor,
                Ts&&...
#if !defined(HPX_COMPUTE_DEVICE_CODE)
                ts
#endif
                ) -> typename hpx::traits::executor_future<Executor,
                typename bulk_then_execute_result<F, Shape, Future,
                    Ts...>::type>::type
            {
#if defined(HPX_COMPUTE_DEVICE_CODE)
                HPX_UNUSED(exec);
                HPX_UNUSED(f);
                HPX_UNUSED(shape);
                HPX_UNUSED(predecessor);
                HPX_ASSERT(false);
                return typename hpx::traits::executor_future<Executor,
                    typename bulk_then_execute_result<F, Shape, Future,
                        Ts...>::type>::type{};
#else
                // result_of_t<F(Shape::value_type, Future)>
                typedef typename then_bulk_function_result<F, Shape, Future,
                    Ts...>::type func_result_type;

                // std::vector<future<func_result_type>>
                typedef std::vector<typename hpx::traits::executor_future<
                    Executor, func_result_type, Ts...>::type>
                    result_type;

                auto func = make_fused_bulk_async_execute_helper<result_type>(
                    exec, std::forward<F>(f), shape,
                    hpx::make_tuple(std::forward<Ts>(ts)...));

                // void or std::vector<func_result_type>
                typedef typename bulk_then_execute_result<F, Shape, Future,
                    Ts...>::type vector_result_type;

                // future<vector_result_type>
                typedef typename hpx::traits::executor_future<Executor,
                    vector_result_type>::type result_future_type;

                typedef typename hpx::traits::detail::shared_state_ptr<
                    vector_result_type>::type shared_state_type;

                typedef typename std::decay<Future>::type future_type;

                shared_state_type p =
                    lcos::detail::make_continuation_exec<vector_result_type>(
                        std::forward<Future>(predecessor),
                        std::forward<BulkExecutor>(exec),
                        [func = std::move(func)](
                            future_type&& predecessor) mutable
                        -> vector_result_type {
                            // use unwrap directly (instead of lazily) to avoid
                            // having to pull in dataflow
                            return hpx::util::unwrap(
                                func(std::move(predecessor)));
                        });

                return hpx::traits::future_access<result_future_type>::create(
                    std::move(p));
#endif
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            HPX_FORCEINLINE static auto call_impl(int, BulkExecutor&& exec,
                F&& f, Shape const& shape, Future&& predecessor, Ts&&... ts)
                -> decltype(exec.bulk_then_execute(std::forward<F>(f), shape,
                    std::forward<Future>(predecessor), std::forward<Ts>(ts)...))
            {
                return exec.bulk_then_execute(std::forward<F>(f), shape,
                    std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            HPX_FORCEINLINE static auto call(BulkExecutor&& exec, F&& f,
                Shape const& shape, Future&& predecessor, Ts&&... ts)
                -> decltype(call_impl(0, std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape,
                    hpx::lcos::make_shared_future(
                        std::forward<Future>(predecessor)),
                    std::forward<Ts>(ts)...))
            {
                return call_impl(0, std::forward<BulkExecutor>(exec),
                    std::forward<F>(f), shape,
                    hpx::lcos::make_shared_future(
                        std::forward<Future>(predecessor)),
                    std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename... Ts>
            struct result
            {
                using type = decltype(call(std::declval<BulkExecutor>(),
                    std::declval<F>(), std::declval<Shape const&>(),
                    std::declval<Future>(), std::declval<Ts>()...));
            };
        };
    }    // namespace detail
    /// \endcond
}}}    // namespace hpx::parallel::execution
