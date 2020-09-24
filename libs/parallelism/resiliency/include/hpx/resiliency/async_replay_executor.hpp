//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2020 Hartmut Kaiser
//  Copyright (c) 2018-2019 Adrian Serio
//  Copyright (c) 2019 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/resiliency/config.hpp>
#include <hpx/resiliency/async_replay.hpp>
#include <hpx/resiliency/resiliency_cpos.hpp>

#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace resiliency { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename Pred, typename F, typename Tuple>
        struct async_replay_executor_helper
          : std::enable_shared_from_this<
                async_replay_executor_helper<Result, Pred, F, Tuple>>
        {
            template <typename Pred_, typename F_, typename Tuple_>
            async_replay_executor_helper(Pred_&& pred, F_&& f, Tuple_&& tuple)
              : pred_(std::forward<Pred_>(pred))
              , f_(std::forward<F_>(f))
              , t_(std::forward<Tuple_>(tuple))
            {
            }

            template <typename Executor, std::size_t... Is>
            typename hpx::parallel::execution::executor_future<Executor,
                Result>::type
            invoke(Executor&& exec, hpx::util::index_pack<Is...>)
            {
                return hpx::parallel::execution::async_execute(
                    std::forward<Executor>(exec), f_, std::get<Is>(t_)...);
            }

            template <typename Executor>
            typename hpx::parallel::execution::executor_future<Executor,
                Result>::type
            call(Executor&& exec, std::size_t n)
            {
                // launch given function asynchronously
                using pack_type =
                    hpx::util::make_index_pack<std::tuple_size<Tuple>::value>;
                using result_type =
                    typename hpx::parallel::execution::executor_future<Executor,
                        Result>::type;

                result_type f = invoke(exec, pack_type{});

                // attach a continuation that will relaunch the task, if
                // necessary
                auto this_ = this->shared_from_this();
                return f.then(hpx::launch::sync,
                    [this_ = std::move(this_),
                        exec = std::forward<Executor>(exec),
                        n](result_type&& f) mutable {
                        if (f.has_exception())
                        {
                            // rethrow abort_replay_exception, if caught
                            auto ex = rethrow_on_abort_replay(f);

                            // execute the task again if an error occurred and
                            // this was not the last attempt
                            if (n != 0)
                            {
                                return this_->call(std::move(exec), n - 1);
                            }

                            // rethrow exception if the number of replays has
                            // been exhausted
                            std::rethrow_exception(ex);
                        }

                        auto&& result = f.get();

                        if (!HPX_INVOKE(this_->pred_, result))
                        {
                            // execute the task again if an error occurred and
                            // this was not the last attempt
                            if (n != 0)
                            {
                                return this_->call(std::move(exec), n - 1);
                            }

                            // throw aborting exception as attempts were
                            // exhausted
                            throw abort_replay_exception();
                        }

                        if (n != 0)
                        {
                            // return result
                            return hpx::make_ready_future(std::move(result));
                        }

                        // throw aborting exception as attempts were
                        // exhausted
                        throw abort_replay_exception();
                    });
            }

            Pred pred_;
            F f_;
            Tuple t_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Pred, typename F, typename Tuple>
        struct async_replay_executor_helper<void, Pred, F, Tuple>
          : std::enable_shared_from_this<
                async_replay_executor_helper<void, Pred, F, Tuple>>
        {
            template <typename Pred_, typename F_, typename Tuple_>
            async_replay_executor_helper(Pred_&&, F_&& f, Tuple_&& tuple)
              : f_(std::forward<F_>(f))
              , t_(std::forward<Tuple_>(tuple))
            {
            }

            template <typename Executor, std::size_t... Is>
            typename hpx::parallel::execution::executor_future<Executor,
                void>::type
            invoke(Executor&& exec, hpx::util::index_pack<Is...>)
            {
                return hpx::parallel::execution::async_execute(
                    std::forward<Executor>(exec), f_, std::get<Is>(t_)...);
            }

            template <typename Executor>
            typename hpx::parallel::execution::executor_future<Executor,
                void>::type
            call(Executor&& exec, std::size_t n)
            {
                // launch given function asynchronously
                using pack_type =
                    hpx::util::make_index_pack<std::tuple_size<Tuple>::value>;
                using result_type =
                    typename hpx::parallel::execution::executor_future<Executor,
                        void>::type;

                result_type f = invoke(exec, pack_type{});

                // attach a continuation that will relaunch the task, if
                // necessary
                auto this_ = this->shared_from_this();
                return f.then(hpx::launch::sync,
                    [this_ = std::move(this_),
                        exec = std::forward<Executor>(exec),
                        n](result_type&& f) mutable {
                        if (f.has_exception())
                        {
                            // rethrow abort_replay_exception, if caught
                            auto ex = rethrow_on_abort_replay(f);

                            // execute the task again if an error occurred and
                            // this was not the last attempt
                            if (n != 0)
                            {
                                return this_->call(std::move(exec), n - 1);
                            }

                            // rethrow exception if the number of replays has
                            // been exhausted
                            std::rethrow_exception(ex);
                        }

                        if (n != 0)
                        {
                            // return result
                            return hpx::make_ready_future();
                        }

                        // throw aborting exception as attempts were
                        // exhausted
                        throw abort_replay_exception();
                    });
            }

            F f_;
            Tuple t_;
        };

        template <typename Result, typename Pred, typename F, typename... Ts>
        std::shared_ptr<async_replay_executor_helper<Result,
            typename std::decay<Pred>::type, typename std::decay<F>::type,
            std::tuple<typename std::decay<Ts>::type...>>>
        make_async_replay_executor_helper(Pred&& pred, F&& f, Ts&&... ts)
        {
            using return_type = async_replay_executor_helper<Result,
                typename std::decay<Pred>::type, typename std::decay<F>::type,
                std::tuple<typename std::decay<Ts>::type...>>;

            return std::make_shared<return_type>(std::forward<Pred>(pred),
                std::forward<F>(f), std::make_tuple(std::forward<Ts>(ts)...));
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f. Verify the result of
    // those invocations using the given predicate \a pred. Repeat launching
    // on error exactly \a n times (except if abort_replay_exception is thrown).
    // clang-format off
    template <typename Executor, typename Pred, typename F, typename... Ts,
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_one_way_executor<Executor>::value ||
            hpx::traits::is_two_way_executor<Executor>::value ||
            hpx::traits::is_threads_executor<Executor>::value
        )>
#else
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_one_way_executor<Executor>::value ||
            hpx::traits::is_two_way_executor<Executor>::value
        )>
#endif
    // clang-format on
    decltype(auto) tag_invoke(async_replay_validate_t, Executor&& exec,
        std::size_t n, Pred&& pred, F&& f, Ts&&... ts)
    {
        using result_type =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;

        auto helper = detail::make_async_replay_executor_helper<result_type>(
            std::forward<Pred>(pred), std::forward<F>(f),
            std::forward<Ts>(ts)...);

        return helper->call(std::forward<Executor>(exec), n);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f. Repeat launching
    // on error exactly \a n times (except if abort_replay_exception is thrown).
    // clang-format off
    template <typename Executor, typename F, typename... Ts,
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_one_way_executor<Executor>::value ||
            hpx::traits::is_two_way_executor<Executor>::value ||
            hpx::traits::is_threads_executor<Executor>::value
        )>
#else
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_one_way_executor<Executor>::value ||
            hpx::traits::is_two_way_executor<Executor>::value
        )>
#endif
    // clang-format on
    decltype(auto) tag_invoke(
        async_replay_t, Executor&& exec, std::size_t n, F&& f, Ts&&... ts)
    {
        using result_type =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;

        auto helper = detail::make_async_replay_executor_helper<result_type>(
            detail::replay_validator{}, std::forward<F>(f),
            std::forward<Ts>(ts)...);

        return helper->call(std::forward<Executor>(exec), n);
    }
}}}    // namespace hpx::resiliency::experimental
