//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file async_base/post.hpp
/// \page hpx::post
/// \headerfile hpx/future.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx {
    // clang-format off

    /// \brief Runs the function \c f asynchronously (potentially in a separate
    ///        thread which might be a part of a thread pool). This is done in
    ///        a fire-and-forget manner, meaning there is no return value or way
    ///        to synchronize with the function execution (it does not return an
    ///        \c hpx::future that would hold the result of that function call).
    ///
    /// \details \c hpx::post is particularly useful when synchronization mechanisms
    ///          as heavyweight as futures are not desired, and instead, more
    ///          lightweight mechanisms like latches or atomic variables are preferred.
    ///          Essentially, the post function enables the launch of a new thread
    ///          without the overhead of creating a future.
    ///
    /// \note \c hpx::post is similar to \c hpx::async but does not return a future.
    ///       This is why there is no way of finding out the result/failure of the
    ///       execution of this function.
    ///
    template <typename F, typename... Ts>
    HPX_FORCEINLINE bool post(F&& f, Ts&&... ts);
    // clang-format on
}    // namespace hpx

#else

#include <hpx/config.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::detail {

    // dispatch point used for post implementations
    template <typename Func, typename Enable = void>
    struct post_dispatch;
}    // namespace hpx::detail

namespace hpx {
    template <typename F, typename... Ts>
    HPX_FORCEINLINE bool post(F&& f, Ts&&... ts)
    {
        return detail::post_dispatch<std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    template <typename F, typename... Ts>
    HPX_DEPRECATED_V(1, 9, "hpx::apply is deprecated, use hpx::post instead")
    HPX_FORCEINLINE bool apply(F&& f, Ts&&... ts)
    {
        return detail::post_dispatch<std::decay_t<F>>::call(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx

#endif
