//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2019 Hartmut Kaiser
//  Copyright (c) 2018-2019 Adrian Serio
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RESILIENCY_DATAFLOW_REPLAY_HPP_2019_FEB_04_0449PM)
#define HPX_RESILIENCY_DATAFLOW_REPLAY_HPP_2019_FEB_04_0449PM

#include <hpx/resiliency/config.hpp>
#include <hpx/resiliency/async_replay.hpp>

#include <hpx/async.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/lcos/future.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace resiliency {

    /// Asynchronously launch given function \a f. Verify the result of
    /// those invocations using the given predicate \a pred. Repeat launching
    /// on error exactly \a n times.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    template <typename Pred, typename F, typename... Ts>
    hpx::future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    dataflow_replay_validate(std::size_t n, Pred&& pred, F&& f, Ts&&... ts)
    {
        return hpx::dataflow(
            hpx::resiliency::functional::async_replay_validate{}, n,
            std::forward<Pred>(pred), std::forward<F>(f),
            std::forward<Ts>(ts)...);
    }

    /// Asynchronously launch given function \a f. Repeat launching on error
    /// exactly \a n times.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    template <typename F, typename... Ts>
    hpx::future<
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
    dataflow_replay(std::size_t n, F&& f, Ts&&... ts)
    {
        return hpx::dataflow(hpx::resiliency::functional::async_replay{}, n,
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}}    // namespace hpx::resiliency

#endif
