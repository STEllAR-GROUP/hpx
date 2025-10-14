//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file source_location.hpp
/// \page HPX_CURRENT_SOURCE_LOCATION, hpx::source_location
/// \headerfile hpx/source_location.hpp

#pragma once

#include <hpx/config/export_definitions.hpp>

#include <cstdint>
#include <iosfwd>

#if defined(HPX_HAVE_CXX20_SOURCE_LOCATION)
#include <source_location>
#endif

namespace hpx {

    /// This contains the location information where \a HPX_ASSERT has been
    /// called
#if defined(HPX_HAVE_CXX20_SOURCE_LOCATION)
    HPX_CORE_MODULE_EXPORT_EXTERN using std::source_location;
#else
    /// The \a source_location class represents certain information about the
    /// source code, such as file names, line numbers, and function names.
    /// Previously, functions that desire to obtain this information about
    /// the call site (for logging, testing, or debugging purposes) must
    /// use macros so that predefined macros like \a __LINE__ and \a __FILE__
    /// are expanded in the context of the caller. The \a source_location class
    /// provides a better alternative.
    /// \a source_location meets the \a DefaultConstructible, \a CopyConstructible,
    /// \a CopyAssignable and \a Destructible requirements. Lvalue of \a
    /// source_location meets the Swappable requirement. Additionally, the following
    /// conditions are true:
    /// - \code std::is_nothrow_move_constructible_v<std::source_location> \endcode
    /// - \code std::is_nothrow_move_assignable_v<std::source_location> \endcode
    /// - \code std::is_nothrow_swappable_v<std::source_location> \endcode
    /// It is intended that source_location has a small size and can be copied
    /// efficiently.
    /// It is unspecified whether the copy/move constructors and the copy/move
    /// assignment operators of \a source_location are trivial and/or constexpr.
    HPX_CORE_MODULE_EXPORT_EXTERN struct source_location
    {
        char const* filename;
        std::uint_least32_t line_number;
        char const* functionname;

        // compatibility with C++20 std::source_location
        /// return the line number represented by this object
        [[nodiscard]] constexpr std::uint_least32_t line() const noexcept
        {
            return line_number;
        }

        /// return the column number represented by this object
        [[nodiscard]] static constexpr std::uint_least32_t column() noexcept
        {
            return 0;
        }

        /// return the file name represented by this object
        [[nodiscard]] constexpr char const* file_name() const noexcept
        {
            return filename;
        }

        /// return the name of the function represented by this object, if any
        [[nodiscard]] constexpr char const* function_name() const noexcept
        {
            return functionname;
        }
    };
#endif

    HPX_CORE_MODULE_EXPORT std::ostream& operator<<(
        std::ostream& os, source_location const& loc);
}    // namespace hpx
