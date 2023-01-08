//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx {

    /// \namespace lcos
    namespace lcos::detail {

        template <typename Result>
        struct future_data;

        template <typename Result>
        struct future_data_base;

        struct future_data_refcnt_base;
    }    // namespace lcos::detail

    /// The class template hpx::future provides a mechanism to access the result
    /// of asynchronous operations:
    ///   - An asynchronous operation (created via hpx::async, hpx::packaged_task,
    ///     or hpx::promise) can provide a hpx::future object to the creator of
    ///     that asynchronous operation.
    ///   - The creator of the asynchronous operation can then use a variety of
    ///     methods to query, wait for, or extract a value from the hpx::future.
    ///     These methods may block if the asynchronous operation has not yet
    ///     provided a value.
    ///   - When the asynchronous operation is ready to send a result to the
    ///     creator, it can do so by modifying shared state (e.g.
    ///     hpx::promise::set_value) that is linked to the creator's hpx::future.
    /// Note that hpx::future references shared state that is not shared with any
    /// other asynchronous return objects (as opposed to hpx::shared_future).
    template <typename R>
    class future;

    /// The class template hpx::shared_future provides a mechanism to access the
    /// result of asynchronous operations, similar to hpx::future, except that
    /// multiple threads are allowed to wait for the same shared state. Unlike
    /// hpx::future, which is only moveable (so only one instance can refer to any
    /// particular asynchronous result), hpx::shared_future is copyable and multiple
    /// shared future objects may refer to the same shared state.
    /// Access to the same shared state from multiple threads is safe if each thread
    /// does it through its own copy of a shared_future object.
    template <typename R>
    class shared_future;

    namespace lcos {

        template <typename R>
        using future HPX_DEPRECATED_V(
            1, 8, "hpx::lcos::future is deprecated. Use hpx::future instead.") =
            hpx::future<R>;

        template <typename R>
        using shared_future HPX_DEPRECATED_V(1, 8,
            "hpx::lcos::shared_future is deprecated. Use hpx::shared_future "
            "instead.") = hpx::shared_future<R>;
    }    // namespace lcos
}    // namespace hpx
