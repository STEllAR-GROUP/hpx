//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <cstddef>

namespace hpx::threads {

    /// Resumes the given processing unit. When the processing unit has been
    /// resumed the returned future will be ready.
    ///
    /// \note Can only be called from an HPX thread. Use
    ///       resume_processing_unit_cb or to resume the processing unit from
    ///       outside HPX. Requires that the pool has
    ///       threads::policies::enable_elasticity set.
    ///
    /// \param virt_core [in] The processing unit on the the pool to be resumed.
    ///                  The processing units are indexed starting from 0.
    ///
    /// \returns A `future<void>` which is ready when the given processing unit
    ///          has been resumed.
    HPX_CORE_EXPORT hpx::future<void> resume_processing_unit(
        thread_pool_base& pool, std::size_t virt_core);

    /// Resumes the given processing unit. Takes a callback as a parameter which
    /// will be called when the processing unit has been resumed.
    ///
    /// \note Requires that the pool has threads::policies::enable_elasticity
    ///       set.
    ///
    /// \param callback  [in] Callback which is called when the processing
    ///                  unit has been suspended.
    /// \param virt_core [in] The processing unit to resume.
    /// \param ec        [in,out] this represents the error status on exit, if this
    ///                  is pre-initialized to \a hpx#throws the function will throw
    ///                  on error instead.
    HPX_CORE_EXPORT void resume_processing_unit_cb(thread_pool_base& pool,
        hpx::function<void(void)> callback, std::size_t virt_core,
        error_code& ec = throws);

    /// Suspends the given processing unit. When the processing unit has been
    /// suspended the returned future will be ready.
    ///
    /// \note Can only be called from an HPX thread. Use
    ///       suspend_processing_unit_cb or to suspend the processing unit from
    ///       outside HPX. Requires that the pool has
    ///       threads::policies::enable_elasticity set.
    ///
    /// \param virt_core [in] The processing unit on the the pool to be
    ///                  suspended. The processing units are indexed starting
    ///                  from 0.
    ///
    /// \returns A `future<void>` which is ready when the given processing unit
    ///          has been suspended.
    ///
    /// \throws hpx::exception if called from outside the HPX runtime.
    HPX_CORE_EXPORT hpx::future<void> suspend_processing_unit(
        thread_pool_base& pool, std::size_t virt_core);

    /// Suspends the given processing unit. Takes a callback as a parameter
    /// which will be called when the processing unit has been suspended.
    ///
    /// \note Requires that the pool has
    ///       threads::policies::enable_elasticity set.
    ///
    /// \param callback  [in] Callback which is called when the processing
    ///                  unit has been suspended.
    /// \param virt_core [in] The processing unit to suspend.
    /// \param ec        [in,out] this represents the error status on exit, if this
    ///                  is pre-initialized to \a hpx#throws the function will throw
    ///                  on error instead.
    HPX_CORE_EXPORT void suspend_processing_unit_cb(
        hpx::function<void(void)> callback, thread_pool_base& pool,
        std::size_t virt_core, error_code& ec = throws);

    /// Resumes the thread pool. When the all OS threads on the thread pool have
    /// been resumed the returned future will be ready.
    ///
    /// \note Can only be called from an HPX thread. Use resume_cb or
    ///       resume_direct to suspend the pool from outside HPX.
    ///
    /// \returns A `future<void>` which is ready when the thread pool has been
    ///          resumed.
    ///
    /// \throws hpx::exception if called from outside the HPX runtime.
    HPX_CORE_EXPORT hpx::future<void> resume_pool(thread_pool_base& pool);

    /// Resumes the thread pool. Takes a callback as a parameter which will be
    /// called when all OS threads on the thread pool have been resumed.
    ///
    /// \param callback [in] called when the thread pool has been resumed.
    /// \param ec       [in,out] this represents the error status on exit, if this
    ///                 is pre-initialized to \a hpx#throws the function will throw
    ///                 on error instead.
    HPX_CORE_EXPORT void resume_pool_cb(thread_pool_base& pool,
        hpx::function<void(void)> callback, error_code& ec = throws);

    /// Suspends the thread pool. When the all OS threads on the thread pool
    /// have been suspended the returned future will be ready.
    ///
    /// \note Can only be called from an HPX thread. Use suspend_cb or
    ///       suspend_direct to suspend the pool from outside HPX. A thread pool
    ///       cannot be suspended from an HPX thread running on the pool itself.
    ///
    /// \returns A `future<void>` which is ready when the thread pool has
    ///          been suspended.
    ///
    /// \throws hpx::exception if called from outside the HPX runtime.
    HPX_CORE_EXPORT hpx::future<void> suspend_pool(thread_pool_base& pool);

    /// Suspends the thread pool. Takes a callback as a parameter which will be
    /// called when all OS threads on the thread pool have been suspended.
    ///
    /// \note A thread pool cannot be suspended from an HPX thread running on
    ///       the pool itself.
    ///
    /// \param callback [in] called when the thread pool has been suspended.
    /// \param ec       [in,out] this represents the error status on exit, if this
    ///                 is pre-initialized to \a hpx#throws the function will throw
    ///                 on error instead.
    ///
    /// \throws hpx::exception if called from an HPX thread which is running
    ///         on the pool itself.
    HPX_CORE_EXPORT void suspend_pool_cb(thread_pool_base& pool,
        hpx::function<void(void)> callback, error_code& ec = throws);
}    // namespace hpx::threads
