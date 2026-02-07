//  Copyright (c) 2007-2025 Hartmut Kaiser
//  Copyright (c) 2025 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>

namespace hpx {

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
    ///     hpx::wait_for_local_termination();
    ///
    ///     return hpx::finalize();
    /// }
    /// \endcode
    ///
    /// \see wait_for_global_termination
    HPX_CORE_EXPORT void wait_for_local_termination();

#if defined(HPX_HAVE_NETWORKING)
    /// \brief Wait for global termination across all localities.
    ///
    /// This function performs distributed termination detection using
    /// Dijkstra's algorithm. It blocks until all localities in the
    /// distributed system have become passive (no pending work).
    ///
    /// The algorithm works by passing a token around all localities in a ring.
    /// Each locality colors itself white or black based on whether it has sent
    /// messages. When a white token completes a full circuit, global
    /// termination is detected.
    ///
    /// \note This function is only available when HPX is built with networking
    ///       support (HPX_WITH_NETWORKING=ON).
    ///
    /// \note This function should typically be called from the root locality
    ///       (locality 0) to coordinate global shutdown.
    ///
    /// \throws hpx::exception if called when the runtime is not in a valid
    ///         state or if networking is not enabled.
    ///
    /// \par Example:
    /// \code
    /// #include <hpx/hpx_init.hpp>
    /// #include <hpx/runtime_local/termination_detection.hpp>
    ///
    /// int hpx_main()
    /// {
    ///     // Launch distributed work across localities
    ///     for (auto const& locality : hpx::find_all_localities())
    ///     {
    ///         hpx::post(locality, []{ /* do work */ });
    ///     }
    ///
    ///     // Wait for global termination across all localities
    ///     if (hpx::get_locality_id() == 0)
    ///     {
    ///         hpx::wait_for_global_termination();
    ///     }
    ///
    ///     return hpx::finalize();
    /// }
    /// \endcode
    ///
    /// \see wait_for_local_termination
    HPX_CORE_EXPORT void wait_for_global_termination();
#endif

}    // namespace hpx
