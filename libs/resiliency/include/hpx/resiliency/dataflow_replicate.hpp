//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2019 Hartmut Kaiser
//  Copyright (c) 2018-2019 Adrian Serio
//  Copyright (c) 2019 Nikunj Gupta
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RESILIENCY_DATAFLOW_REPLICATE_HPP_2018_OCT_20_0548PM)
#define HPX_RESILIENCY_DATAFLOW_REPLICATE_HPP_2018_OCT_20_0548PM

#include <hpx/resiliency/async_replicate.hpp>
#include <hpx/resiliency/config.hpp>

#include <hpx/async.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/lcos/future.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace resiliency {

    /// Launch given function \a f exactly \a n times. Run all the valid
    /// results against a user provided voting function.
    /// Return the valid output.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    template <typename Vote, typename Pred, typename F, typename... Ts>
    hpx::future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    dataflow_replicate_vote_validate(
        std::size_t n, Vote&& vote, Pred&& pred, F&& f, Ts&&... ts)
    {
        return hpx::dataflow(
            hpx::resiliency::functional::async_replicate_vote_validate{}, n,
            std::forward<Vote>(vote), std::forward<Pred>(pred),
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }

    /// Launch given function \a f exactly \a n times. Run all the valid
    /// results against a user provided voting function.
    /// Return the valid output.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    template <typename Vote, typename F, typename... Ts>
    hpx::future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    dataflow_replicate_vote(std::size_t n, Vote&& vote, F&& f, Ts&&... ts)
    {
        return hpx::dataflow(
            hpx::resiliency::functional::async_replicate_vote{}, n,
            std::forward<Vote>(vote), std::forward<F>(f),
            std::forward<Ts>(ts)...);
    }

    /// Launch given function \a f exactly \a n times. Verify the result of
    /// those invocations using the given predicate \a pred. Return the first
    /// valid result.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    template <typename Pred, typename F, typename... Ts>
    hpx::future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    dataflow_replicate_validate(std::size_t n, Pred&& pred, F&& f, Ts&&... ts)
    {
        return hpx::dataflow(
            hpx::resiliency::functional::async_replicate_validate{}, n,
            std::forward<Pred>(pred), std::forward<F>(f),
            std::forward<Ts>(ts)...);
    }

    /// Launch given function \a f exactly \a n times. Return the first
    /// valid result.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    template <typename F, typename... Ts>
    hpx::future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    dataflow_replicate(std::size_t n, F&& f, Ts&&... ts)
    {
        return hpx::dataflow(hpx::resiliency::functional::async_replicate{}, n,
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}}    // namespace hpx::resiliency

#endif
