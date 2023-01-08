//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx::agas {

    enum class service_mode
    {
        invalid = -1,
        bootstrap = 0,
        hosted = 1
    };

#define HPX_AGAS_SERVICE_MODE_UNSCOPED_ENUM_DEPRECATION_MSG                    \
    "The unscoped hpx::agas::service_mode names are deprecated. Please use "   \
    "hpx::agas::service_mode::state instead."

    HPX_DEPRECATED_V(1, 9, HPX_AGAS_SERVICE_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr service_mode service_mode_invalid = service_mode::invalid;
    HPX_DEPRECATED_V(1, 9, HPX_AGAS_SERVICE_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr service_mode service_mode_bootstrap =
        service_mode::bootstrap;
    HPX_DEPRECATED_V(1, 9, HPX_AGAS_SERVICE_MODE_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr service_mode service_mode_hosted = service_mode::hosted;

#undef HPX_AGAS_SERVICE_MODE_UNSCOPED_ENUM_DEPRECATION_MSG

}    // namespace hpx::agas
