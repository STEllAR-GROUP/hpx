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

namespace hpx { namespace resiliency { namespace experimental {
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

                using future_type =
                    typename hpx::traits::executor_future<Executor,
                        result_type>::type;

                // launch given function n times
                auto func = [f = std::forward<F>(f),
                                t = hpx::make_tuple(std::forward<Ts>(ts)...)](
                                std::size_t) mutable -> result_type {
                    // ignore argument (invocation count of bulk_execute)
                    return hpx::util::invoke_fused(f, t);
                };

                std::vector<future_type> results =
                    hpx::parallel::execution::bulk_async_execute(
                        std::forward<Executor>(exec), std::move(func), n);

                // wait for all threads to finish executing and return the first
                // result that passes the predicate, properly handle exceptions
                // do not schedule new thread for the lambda
                return hpx::dataflow(
                    hpx::launch::sync,
                    [pred = std::forward<Pred>(pred),
                        vote = std::forward<Vote>(vote),
                        n](std::vector<future_type>&& results) mutable
                    -> result_type {
                        // Store all valid results
                        std::vector<result_type> valid_results;
                        valid_results.reserve(n);

                        std::exception_ptr ex;

                        for (auto&& f : std::move(results))
                        {
                            if (f.has_exception())
                            {
                                // rethrow abort_replicate_exception, if caught
                                ex = detail::rethrow_on_abort_replicate(f);
                            }
                            else
                            {
                                auto&& result = f.get();
                                if (HPX_INVOKE(pred, result))
                                {
                                    valid_results.emplace_back(
                                        std::move(result));
                                }
                            }
                        }

                        if (!valid_results.empty())
                        {
                            return HPX_INVOKE(std::forward<Vote>(vote),
                                std::move(valid_results));
                        }

                        if (bool(ex))
                        {
                            std::rethrow_exception(ex);
                        }

                        // throw aborting exception no correct results ere produced
                        throw abort_replicate_exception{};
                    },
                    std::move(results));
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
                using future_type =
                    typename hpx::traits::executor_future<Executor, void>::type;

                // launch given function n times
                auto func = [f = std::forward<F>(f),
                                t = hpx::make_tuple(std::forward<Ts>(ts)...)](
                                std::size_t) mutable {
                    // ignore argument (invocation count of bulk_execute)
                    hpx::util::invoke_fused(f, t);
                };

                std::vector<future_type> results =
                    hpx::parallel::execution::bulk_async_execute(
                        std::forward<Executor>(exec), std::move(func), n);

                // wait for all threads to finish executing and return the first
                // result that passes the predicate, properly handle exceptions
                // do not schedule new thread for the lambda
                return hpx::dataflow(
                    hpx::launch::sync,
                    [](std::vector<future_type>&& results) mutable -> void {
                        std::exception_ptr ex;

                        std::size_t count = 0;
                        for (auto&& f : std::move(results))
                        {
                            if (f.has_exception())
                            {
                                // rethrow abort_replicate_exception, if caught
                                ex = detail::rethrow_on_abort_replicate(f);
                            }
                            else
                            {
                                ++count;
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

                        // throw aborting exception if no correct results were produced
                        throw abort_replicate_exception{};
                    },
                    std::move(results));
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
    decltype(auto) tag_invoke(async_replicate_vote_validate_t, Executor&& exec,
        std::size_t n, Vote&& vote, Pred&& pred, F&& f, Ts&&... ts)
    {
        using result_type =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;

        return detail::async_replicate_vote_validate_executor<
            result_type>::call(std::forward<Executor>(exec), n,
            std::forward<Vote>(vote), std::forward<Pred>(pred),
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times. Verify
    // the result of those invocations using the given predicate \a pred. Run
    // all the valid results against a user provided voting function.
    // Return the valid output.
    // clang-format off
    template <typename Executor, typename Vote, typename F, typename... Ts,
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
    decltype(auto) tag_invoke(async_replicate_vote_t, Executor&& exec,
        std::size_t n, Vote&& vote, F&& f, Ts&&... ts)
    {
        using result_type =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;

        return detail::async_replicate_vote_validate_executor<
            result_type>::call(std::forward<Executor>(exec), n,
            std::forward<Vote>(vote), detail::replicate_validator{},
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times. Verify
    // the result of those invocations using the given predicate \a pred.
    // Return the first valid result.
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
    decltype(auto) tag_invoke(async_replicate_validate_t, Executor&& exec,
        std::size_t n, Pred&& pred, F&& f, Ts&&... ts)
    {
        using result_type =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;

        return detail::async_replicate_vote_validate_executor<
            result_type>::call(std::forward<Executor>(exec), n,
            detail::replicate_voter{}, std::forward<Pred>(pred),
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times. Verify
    // the result of those invocations by checking for exception.
    // Return the first valid result.
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
        async_replicate_t, Executor&& exec, std::size_t n, F&& f, Ts&&... ts)
    {
        using result_type =
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type;

        return detail::async_replicate_vote_validate_executor<
            result_type>::call(std::forward<Executor>(exec), n,
            detail::replicate_voter{}, detail::replicate_validator{},
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}}}    // namespace hpx::resiliency::experimental
