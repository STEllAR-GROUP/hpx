////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime_configuration/runtime_mode.hpp>

#include <cstddef>
#include <string>

namespace hpx {

    namespace strings {

        inline constexpr char const* const runtime_mode_names[] = {
            "invalid",    // -1
            "console",    // 0
            "worker",     // 1
            "connect",    // 2
            "local",      // 3
            "default",    // 4
        };
    }

    char const* get_runtime_mode_name(runtime_mode state) noexcept
    {
        if (state < runtime_mode::invalid || state > runtime_mode::last)
            return "invalid (value out of bounds)";
        return strings::runtime_mode_names[static_cast<int>(state) + 1];
    }

    runtime_mode get_runtime_mode_from_name(std::string const& mode)
    {
        for (std::size_t i = 0;
            static_cast<runtime_mode>(i) <= runtime_mode::last; ++i)
        {
            if (mode == strings::runtime_mode_names[i])
                return static_cast<runtime_mode>(i - 1);
        }
        return runtime_mode::invalid;
    }
}    // namespace hpx
