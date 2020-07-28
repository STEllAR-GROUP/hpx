//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file runtime_mode.hpp

#pragma once

#include <hpx/config.hpp>

#include <string>

namespace hpx {
    /// A HPX runtime can be executed in two different modes: console mode
    /// and worker mode.
    enum class runtime_mode
    {
        invalid = -1,
        console = 0,     ///< The runtime is the console locality
        worker = 1,      ///< The runtime is a worker locality
        connect = 2,     ///< The runtime is a worker locality
                         ///< connecting late
        local = 3,       ///< The runtime is fully local
        default_ = 4,    ///< The runtime mode will be determined
                         ///< based on the command line arguments
        last
    };

#if defined(HPX_HAVE_UNSCOPED_ENUM_COMPATIBILITY)
#define HPX_RUNTIME_MODE_UNSCOPED_ENUM_DEPRECATION_MSG                         \
    "The unscoped runtime_mode names are deprecated. Please use "              \
    "runtime_mode::mode instead."

    HPX_DEPRECATED_V(1, 5, HPX_RUNTIME_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr runtime_mode runtime_mode_invalid = runtime_mode::invalid;
    HPX_DEPRECATED_V(1, 5, HPX_RUNTIME_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr runtime_mode runtime_mode_console = runtime_mode::console;
    HPX_DEPRECATED_V(1, 5, HPX_RUNTIME_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr runtime_mode runtime_mode_worker = runtime_mode::worker;
    HPX_DEPRECATED_V(1, 5, HPX_RUNTIME_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr runtime_mode runtime_mode_connect = runtime_mode::connect;
    HPX_DEPRECATED_V(1, 5, HPX_RUNTIME_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr runtime_mode runtime_mode_local = runtime_mode::local;
    HPX_DEPRECATED_V(1, 5, HPX_RUNTIME_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr runtime_mode runtime_mode_default = runtime_mode::default_;
    HPX_DEPRECATED_V(1, 5, HPX_RUNTIME_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    static constexpr runtime_mode runtime_mode_last = runtime_mode::last;
#undef HPX_RUNTIME_MODE_UNSCOPED_ENUM_DEPRECATION_MSG
#endif

    /// Get the readable string representing the name of the given runtime_mode
    /// constant.
    HPX_EXPORT char const* get_runtime_mode_name(runtime_mode state);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns the internal representation (runtime_mode constant) from
    /// the readable string representing the name.
    ///
    /// This represents the internal representation from the readable string
    /// representing the name.
    ///
    /// \param mode this represents the runtime mode
    HPX_EXPORT runtime_mode get_runtime_mode_from_name(std::string const& mode);
}    // namespace hpx
