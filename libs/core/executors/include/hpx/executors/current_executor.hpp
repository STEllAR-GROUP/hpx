//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/executors/parallel_executor.hpp>

namespace hpx::parallel::execution {

    using current_executor HPX_DEPRECATED_V(1, 9,
        "hpx::parallel::execution::current_executor is deprecated, use "
        "hpx::execution::parallel_executor instead") =
        hpx::execution::parallel_executor;
}    // namespace hpx::parallel::execution

namespace hpx::threads {

    /// Returns a reference to the executor that was used to create the given
    /// thread.
    ///
    /// \throws If <code>&ec != &throws</code>, never throws, but will set \a ec
    ///         to an appropriate value when an error occurs. Otherwise, this
    ///         function will throw an \a hpx#exception with an error code of
    ///         \a hpx#error#yield_aborted if it is signaled with \a
    ///            wait_aborted.
    ///         If called outside of a HPX-thread, this function will throw an
    ///         \a hpx#exception with an error code of \a
    ///            hpx#error#null_thread_id.
    ///         If this function is called while the thread-manager is not
    ///         running, it will throw an \a hpx#exception with an error code of
    ///         \a hpx#error#invalid_status.
    ///
    HPX_CORE_EXPORT hpx::execution::parallel_executor get_executor(
        thread_id_type const& id, error_code& ec = throws);
}    // namespace hpx::threads

namespace hpx::this_thread {

    /// Returns a reference to the executor that was used to create the current
    /// thread.
    ///
    /// \throws If <code>&ec != &throws</code>, never throws, but will set \a ec
    ///         to an appropriate value when an error occurs. Otherwise, this
    ///         function will throw an \a hpx#exception with an error code of
    ///         \a hpx#error#yield_aborted if it is signaled with \a
    ///            wait_aborted.
    ///         If called outside of a HPX-thread, this function will throw an
    ///         \a hpx#exception with an error code of \a
    ///            hpx#error#null_thread_id.
    ///         If this function is called while the thread-manager is not
    ///         running, it will throw an \a hpx#exception with an error code of
    ///         \a hpx#error#invalid_status.
    ///
    HPX_CORE_EXPORT hpx::execution::parallel_executor get_executor(
        error_code& ec = throws);
}    // namespace hpx::this_thread
