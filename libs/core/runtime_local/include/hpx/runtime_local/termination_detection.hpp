//  Copyright (c) 2007-2025 Hartmut Kaiser
//  Copyright (c) 2025 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>

namespace hpx { namespace local {

    /// \brief Wait for all local HPX threads to complete execution.
    ///
    /// This function blocks the calling thread until all HPX worker threads
    /// on the current locality have completed their work and become idle.
    /// It performs local termination detection by waiting for the thread
    /// manager to drain all pending tasks.
    ///
    /// \note This function should be called from an HPX thread or after
    ///       the HPX runtime has been initialized.
    ///
    /// \throws hpx::exception if called when the runtime is not in a valid
    ///         state (e.g., before initialization or after shutdown).
    ///
    /// \par Example:
    /// \code
    /// #include <hpx/hpx_init.hpp>
    /// #include <hpx/runtime_local/termination_detection.hpp>
    ///
    /// int hpx_main()
    /// {
    ///     // Launch some asynchronous work
    ///     for (int i = 0; i < 100; ++i)
    ///     {
    ///         hpx::post([]{ /* do work */ });
    ///     }
    ///
    ///     // Wait for all local threads to complete
    ///     hpx::local::termination_detection();
    ///
    ///     return hpx::finalize();
    /// }
    /// \endcode
    HPX_CORE_EXPORT void termination_detection();

}}    // namespace hpx::local
