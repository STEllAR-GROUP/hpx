//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2020 Hartmut Kaiser
//  Copyright (c) 2018-2019 Adrian Serio
//  Copyright (c) 2019-2020 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/resiliency/resiliency_cpos.hpp>
#include <hpx/resiliency/util.hpp>

#include <hpx/assert.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/async_local.hpp>

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hpx { namespace resiliency { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Vote, typename Pred, typename Action, typename... Ts>
        hpx::future<typename hpx::util::detail::invoke_deferred_result<Action,
            hpx::naming::id_type, Ts...>::type>
        async_replicate_vote_validate(
            const std::vector<hpx::naming::id_type>& ids, Vote&& vote,
            Pred&& pred, Action&& action, Ts&&... ts)
        {
            using result_type =
                typename hpx::util::detail::invoke_deferred_result<Action,
                    hpx::naming::id_type, Ts...>::type;

            // launch given function n times
            std::vector<hpx::future<result_type>> results;
            results.reserve(ids.size());

            for (std::size_t i = 0; i != ids.size(); ++i)
            {
                results.emplace_back(hpx::async(action, ids.at(i), ts...));
            }

            // wait for all threads to finish executing and return the first
            // result that passes the predicate, properly handle exceptions
            return hpx::dataflow(
                hpx::launch::
                    sync,    // do not schedule new thread for the lambda
                [pred = std::forward<Pred>(pred),
                    vote = std::forward<Vote>(vote), ids](
                    std::vector<hpx::future<result_type>>&& results) mutable
                -> result_type {
                    // Store all valid results
                    std::vector<result_type> valid_results;
                    valid_results.reserve(ids.size());

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
                            if (hpx::util::invoke(pred, result))
                            {
                                valid_results.emplace_back(std::move(result));
                            }
                        }
                    }

                    if (!valid_results.empty())
                    {
                        return hpx::util::invoke(
                            std::forward<Vote>(vote), std::move(valid_results));
                    }

                    if (bool(ex))
                        std::rethrow_exception(ex);

                    // throw aborting exception no correct results ere produced
                    throw abort_replicate_exception{};
                },
                std::move(results));
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times on \a n
    // different localities, where \a n is the size of the vector of \a ids.
    // Verify the result of those invocations using the given predicate \a pred.
    // Run all the valid results against a user provided voting function.
    // Return the valid output.
    template <typename Vote, typename Pred, typename Action, typename... Ts>
    hpx::future<typename hpx::util::detail::invoke_deferred_result<Action,
        hpx::naming::id_type, Ts...>::type>
    tag_invoke(async_replicate_vote_validate_t,
        const std::vector<hpx::naming::id_type>& ids, Vote&& vote, Pred&& pred,
        Action&& action, Ts&&... ts)
    {
        HPX_ASSERT(ids.size() > 0);

        return detail::async_replicate_vote_validate(ids,
            std::forward<Vote>(vote), std::forward<Pred>(pred),
            std::forward<Action>(action), std::forward<Ts>(ts)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times on \a n
    // different localities, where \a n is the size of the vector of \a ids.
    // Verify the result of those invocations using the given predicate \a pred.
    // Run all the valid results against a user provided voting function.
    // Return the valid output.
    template <typename Vote, typename Action, typename... Ts>
    hpx::future<typename hpx::util::detail::invoke_deferred_result<Action,
        hpx::naming::id_type, Ts...>::type>
    tag_invoke(async_replicate_vote_t,
        const std::vector<hpx::naming::id_type>& ids, Vote&& vote,
        Action&& action, Ts&&... ts)
    {
        HPX_ASSERT(ids.size() > 0);

        return detail::async_replicate_vote_validate(ids,
            std::forward<Vote>(vote), detail::replicate_validator{},
            std::forward<Action>(action), std::forward<Ts>(ts)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times on \a n
    // different localities, where \a n is the size of the vector of \a ids.
    // Verify the result of those invocations using the given predicate \a pred.
    // Run all the valid results against a user provided voting function.
    // Return the valid output.
    template <typename Pred, typename Action, typename... Ts>
    hpx::future<typename hpx::util::detail::invoke_deferred_result<Action,
        hpx::naming::id_type, Ts...>::type>
    tag_invoke(async_replicate_validate_t,
        const std::vector<hpx::naming::id_type>& ids, Pred&& pred,
        Action&& action, Ts&&... ts)
    {
        HPX_ASSERT(ids.size() > 0);

        return detail::async_replicate_vote_validate(ids,
            detail::replicate_voter{}, std::forward<Pred>(pred),
            std::forward<Action>(action), std::forward<Ts>(ts)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronously launch given function \a f exactly \a n times on \a n
    // different localities, where \a n is the size of the vector of \a ids.
    // Verify the result of those invocations using the given predicate \a pred.
    // Run all the valid results against a user provided voting function.
    // Return the valid output.
    template <typename Action, typename... Ts>
    hpx::future<typename hpx::util::detail::invoke_deferred_result<Action,
        hpx::naming::id_type, Ts...>::type>
    tag_invoke(async_replicate_t, const std::vector<hpx::naming::id_type>& ids,
        Action&& action, Ts&&... ts)
    {
        HPX_ASSERT(ids.size() > 0);

        return detail::async_replicate_vote_validate(ids,
            detail::replicate_voter{}, detail::replicate_validator{},
            std::forward<Action>(action), std::forward<Ts>(ts)...);
    }

}}}    // namespace hpx::resiliency::experimental

#endif
