//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assertion/current_function.hpp>
#include <hpx/assertion/export_definitions.hpp>

#include <cstdint>
#include <iosfwd>
#include <ostream>

#if defined(HPX_HAVE_CXX20_SOURCE_LOCATION)
#include <source_location>
#endif

namespace hpx {

    /// This contains the location information where \a HPX_ASSERT has been
    /// called
#if defined(HPX_HAVE_CXX20_SOURCE_LOCATION)
    using std::source_location;
#else
    struct source_location
    {
        const char* filename;
        std::uint_least32_t line_number;
        const char* functionname;

        // compatibility with C++20 std::source_location
        constexpr std::uint_least32_t line() const noexcept
        {
            return line_number;
        }

        constexpr std::uint_least32_t column() const noexcept
        {
            return 0;
        }

        constexpr const char* file_name() const noexcept
        {
            return filename;
        }

        constexpr const char* function_name() const noexcept
        {
            return functionname;
        }
    };
#endif

    HPX_CORE_ASSERTION_EXPORT std::ostream& operator<<(
        std::ostream& os, source_location const& loc);
}    // namespace hpx

namespace hpx::assertion {

    using source_location HPX_DEPRECATED_V(1, 8,
        "hpx::assertion::source_location is deprecated, use "
        "hpx::source_location instead") = hpx::source_location;
}
