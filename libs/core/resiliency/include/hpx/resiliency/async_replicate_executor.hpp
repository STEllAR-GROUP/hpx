//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2023 Hartmut Kaiser
//  Copyright (c) 2018-2019 Adrian Serio
//  Copyright (c) 2019 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/resiliency/config.hpp>
#include <hpx/resiliency/async_replicate.hpp>
#include <hpx/resiliency/resiliency_cpos.hpp>

#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/futures.hpp>

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hpx::resiliency::experimental {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct async_replicate_vote_validate_executor
        {
            template <typename Executor, typename Vote, typename Pred,
                typename F, typename... Ts>
            static typename hpx::traits::executor_future<Executor, Result>::type
            call(Executor&& exec, std::size_t n, Vote&& vote, Pred&& pred,
                F&& f, Ts&&... ts)
            {
                using result_type =
                    typename hpx::util::detail::invoke_deferred_result<F,
                        Ts...>::type;

                // launch given function n times
                auto func = [f = HPX_FORWARD(F, f),
                                t = hpx::make_tuple(HPX_FORWARD(Ts, ts)...)](
                                std::size_t) mutable -> result_type {
                    // ignore argument (invocation count of bulk_execute)
                    return hpx::invoke_fused(f, t);
                };

                auto&& results = hpx::parallel::execution::bulk_async_execute(
                    HPX_FORWARD(Executor, exec), HPX_MOVE(func), n);

                // wait for all threads to finish executing and return the first
                // result that passes the predicate, properly handle exceptions
                // do not schedule new thread for the lambda
                return hpx::dataflow(
                    hpx::launch::sync,
                    [n, pred = HPX_FORWARD(Pred, pred),
                        vote = HPX_FORWARD(Vote, vote)](
                        auto&& results) mutable -> result_type {
                        // Store all valid results
                        std::vector<result_type> valid_results;
                        valid_results.reserve(n);

                        std::exception_ptr ex;

                        // clang-format off
                        if constexpr (
                            hpx::traits::is_future_v<decltype(results)>)
                        // clang-format on
                        {
                            if (results.has_exception())
                            {
                                // rethrow abort_replicate_exception, if caught
                                ex =
                                    detail::rethrow_on_abort_replicate(results);
                            }
                            else
                            {
                                auto&& result = results.get();
                                if (HPX_INVOKE(pred, result))
                                {
                                    valid_results.emplace_back(
                                        HPX_MOVE(result));
                                }
                            }
                        }
                        else
                        {
                            for (auto&& f : HPX_MOVE(results))
                            {
                                if (f.has_exception())
                                {
                                    // rethrow abort_replicate_exception, if
                                    // caught
                                    ex = detail::rethrow_on_abort_replicate(f);
                                }
                                else
                                {
                                    auto&& result = f.get();
                                    if (HPX_INVOKE(pred, result))
                                    {
                                        valid_results.emplace_back(
                                            HPX_MOVE(result));
                                    }
                                }
                            }
                        }

                        if (!valid_results.empty())
                        {
                            return hpx::invoke(HPX_FORWARD(Vote, vote),
                                HPX_MOVE(valid_results));
                        }

                        if (bool(ex))
                        {
                            std::rethrow_exception(ex);
                        }

                        // throw aborting exception no correct results ere
                        // produced
                        throw abort_replicate_exception{};
                    },
                    HPX_MOVE(results));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <>
        struct async_replicate_vote_validate_executor<void>
        {
            template <typename Executor, typename Vote, typename Pred,
                typename F, typename... Ts>
            static typename hpx::traits::executor_future<Executor, void>::type
            call(Executor&& exec, std::size_t n, Vote&&, Pred&&, F&& f,
                Ts&&... ts)
            {
                // launch given function n times
                auto func = [f = HPX_FORWARD(F, f),
                                t = hpx::make_tuple(HPX_FORWARD(Ts, ts)...)](
                                std::size_t) mutable {
                    // ignore argument (invocation count of bulk_execute)
                    hpx::invoke_fused(f, t);

                    // return non-void result to force executor into providing a
                    // future for each invocation (returning void might optimize
                    // bulk_async_execute to return just a single future)
                    return 0;
                };

                auto&& results = hpx::parallel::execution::bulk_async_execute(
                    HPX_FORWARD(Executor, exec), HPX_MOVE(func), n);

                // wait for all threads to finish executing and return the first
                // result that passes the predicate, properly handle exceptions
                // do not schedule new thread for the lambda
                return hpx::dataflow(
                    hpx::launch::sync,
                    [](auto&& results) mutable -> void {
                        std::exception_ptr ex;

                        std::size_t count = 0;
                        // clang-format off
                        if constexpr (hpx::traits::is_future_v<
                                          decltype(results)>)
                        // clang-format on
                        {
                            if (results.has_exception())
                            {
                                // rethrow abort_replicate_exception, if caught
                                ex =
                                    detail::rethrow_on_abort_replicate(results);
                            }
                            else
                            {
                                ++count;
                            }
                        }
                        else
                        {
                            for (auto&& f : HPX_MOVE(results))
                            {
                                if (f.has_exception())
                                {
                                    // rethrow abort_replicate_exception, if
                                    // caught
                                    ex = detail::rethrow_on_abort_replicate(f);
                                }
                                else
                                {
                                    ++count;
                                }
                            }
                        }

                        if (count != 0)
                        {
                            return;
                        }

                        if (bool(ex))
                        {
                            std::rethrow_exception(ex);
                        }

                        // throw aborting exception if no correct results were
                        // produced
                        throw abort_replicate_exception{};
                    },
                    HPX_MOVE(results));
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times. Verify
    // the result of those invocations using the given predicate \a pred.
    // Run all the valid results against a user provided voting function.
    // Return the valid output.
    // clang-format off
    template <typename Executor, typename Vote, typename Pred, typename F,
        typename... Ts,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_one_way_executor_v<Executor> ||
            hpx::traits::is_two_way_executor_v<Executor>
        )>
    // clang-format on
    decltype(auto) tag_invoke(async_replicate_vote_validate_t, Executor&& exec,
        std::size_t n, Vote&& vote, Pred&& pred, F&& f, Ts&&... ts)
    {
        using result_type =
            hpx::util::detail::invoke_deferred_result_t<F, Ts...>;

        return detail::async_replicate_vote_validate_executor<
            result_type>::call(HPX_FORWARD(Executor, exec), n,
            HPX_FORWARD(Vote, vote), HPX_FORWARD(Pred, pred), HPX_FORWARD(F, f),
            HPX_FORWARD(Ts, ts)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times. Verify
    // the result of those invocations using the given predicate \a pred. Run
    // all the valid results against a user provided voting function.
    // Return the valid output.
    // clang-format off
    template <typename Executor, typename Vote, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_one_way_executor_v<Executor> ||
            hpx::traits::is_two_way_executor_v<Executor>
        )>
    // clang-format on
    decltype(auto) tag_invoke(async_replicate_vote_t, Executor&& exec,
        std::size_t n, Vote&& vote, F&& f, Ts&&... ts)
    {
        using result_type =
            hpx::util::detail::invoke_deferred_result_t<F, Ts...>;

        return detail::async_replicate_vote_validate_executor<
            result_type>::call(HPX_FORWARD(Executor, exec), n,
            HPX_FORWARD(Vote, vote), detail::replicate_validator{},
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times. Verify
    // the result of those invocations using the given predicate \a pred.
    // Return the first valid result.
    // clang-format off
    template <typename Executor, typename Pred, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_one_way_executor_v<Executor> ||
            hpx::traits::is_two_way_executor_v<Executor>
        )>
    // clang-format on
    decltype(auto) tag_invoke(async_replicate_validate_t, Executor&& exec,
        std::size_t n, Pred&& pred, F&& f, Ts&&... ts)
    {
        using result_type =
            hpx::util::detail::invoke_deferred_result_t<F, Ts...>;

        return detail::async_replicate_vote_validate_executor<
            result_type>::call(HPX_FORWARD(Executor, exec), n,
            detail::replicate_voter{}, HPX_FORWARD(Pred, pred),
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times. Verify
    // the result of those invocations by checking for exception.
    // Return the first valid result.
    // clang-format off
    template <typename Executor, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_one_way_executor_v<Executor> ||
            hpx::traits::is_two_way_executor_v<Executor>
        )>
    // clang-format on
    decltype(auto) tag_invoke(
        async_replicate_t, Executor&& exec, std::size_t n, F&& f, Ts&&... ts)
    {
        using result_type =
            hpx::util::detail::invoke_deferred_result_t<F, Ts...>;

        return detail::async_replicate_vote_validate_executor<
            result_type>::call(HPX_FORWARD(Executor, exec), n,
            detail::replicate_voter{}, detail::replicate_validator{},
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::resiliency::experimental
