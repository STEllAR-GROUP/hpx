////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <memory>
#include <string>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace hpx::util {

    /// Parse a sed command.
    ///
    /// \param input    [in] The content to parse.
    /// \param search   [out] If the parsing is successful, this string is set to
    ///                 the search expression.
    /// \param replace  [out] If the parsing is successful, this string is set to
    ///                 the replace expression.
    ///
    /// \returns \a true if the parsing was successful, false otherwise.
    ///
    /// \note Currently, only supports search and replace syntax (s/search/replace/)
    HPX_CORE_EXPORT bool parse_sed_expression(
        std::string const& input, std::string& search, std::string& replace);

    /// An unary function object which applies a sed command to its subject and
    /// returns the resulting string.
    ///
    /// \note Currently, only supports search and replace syntax (s/search/replace/)
    struct HPX_CORE_EXPORT sed_transform
    {
    private:
        struct command;

        std::shared_ptr<command> command_;

    public:
        sed_transform(std::string const& search, std::string replace);

        explicit sed_transform(std::string const& expression);

        std::string operator()(std::string const& input) const;

        explicit operator bool() const noexcept
        {
            // avoid compiler warning about conversion to bool
            return command_ ? true : false;
        }

        [[nodiscard]] bool operator!() const noexcept
        {
            return !command_;
        }
    };
}    // namespace hpx::util

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif
