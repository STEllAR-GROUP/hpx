//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file async.hpp

#pragma once

#include <hpx/config.hpp>

#include <type_traits>
#include <utility>

namespace hpx::detail {

    // dispatch point used for async implementations
    template <typename Func, typename Enable = void>
    struct async_dispatch;
}    // namespace hpx::detail

namespace hpx {

    /// The function template \a async runs the function \a f asynchronously
    /// (potentially in a separate thread which might be a part of a thread
    /// pool) and returns an \a hpx::future that will eventually hold the result
    /// of that function call. If no policy is defined, \a async behaves as if
    /// it is called with policy being
    /// \a hpx::launch::async | hpx::launch::deferred. Otherwise, it calls a
    /// function \a f with arguments \a ts according to a specific launch
    /// policy.
    ///     - If the async flag is set (i.e. (policy & hpx::launch::async) !=
    ///       0), then async executes the callable object f on a new thread of
    ///       execution (with all thread-locals initialized) as if spawned by
    ///       hpx::thread(std::forward<F>(f), std::forward<Ts>(ts)...), except
    ///       that if the function f returns a value or throws an exception, it
    ///       is stored in the shared state accessible through the hpx::future
    ///       that async returns to the caller.
    ///     - If the deferred flag is set (i.e. (policy & hpx::launch::deferred)
    ///       != 0), then async converts f and ts... the same way as by
    ///       hpx::thread constructor, but does not spawn a new thread of
    ///       execution. Instead, lazy evaluation is performed: the first call
    ///       to a non-timed wait function on the hpx::future that async
    ///       returned to the caller will cause the copy of f to be invoked (as
    ///       an rvalue) with the copies of ts... (also passed as rvalues) in
    ///       the current thread (which does not have to be the thread that
    ///       originally called hpx::async). The result or exception is placed
    ///       in the shared state associated with the future and only then it is
    ///       made ready. All further accesses to the same hpx::future will
    ///       return the result immediately.
    ///     - If neither hpx::launch::async nor hpx::launch::deferred, nor any
    ///       implementation-defined policy flag is set in policy, the behavior
    ///       is undefined.
    ///
    ///   If more than one flag is set, it is implementation-defined which
    ///   policy is selected. For the default (both the hpx::launch::async and
    ///   hpx::launch::deferred flags are set in policy), standard recommends
    ///   (but doesn't require) utilizing available concurrency, and deferring
    ///   any additional tasks.
    ///
    /// In any case, the call to hpx::async synchronizes-with (as defined in
    /// std::memory_order) the call to f, and the completion of f is
    /// sequenced-before making the shared state ready. If the async policy is
    /// chosen, the associated thread completion synchronizes-with the
    /// successful return from the first function that is waiting on the shared
    /// state, or with the return of the last function that releases the shared
    /// state, whichever comes first. If std::decay<Function>::type or each type
    /// in std::decay<Ts>::type is not constructible from its corresponding
    /// argument, the program is ill-formed.
    ///
    /// \param f     Callable object to call
    /// \param ts... parameters to pass to f
    ///
    /// \returns hpx::future referring to the shared state created by this call
    ///          to \a hpx::async.
    ///
    template <typename F, typename... Ts>
    HPX_FORCEINLINE decltype(auto) async(F&& f, Ts&&... ts)
    {
        return detail::async_dispatch<std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx
