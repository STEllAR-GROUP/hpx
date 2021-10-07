// manipulator.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/logging/message.hpp>
#include <hpx/modules/format.hpp>

#include <boost/utility/string_ref.hpp>

#include <iosfwd>
#include <string>

namespace hpx { namespace util { namespace logging {

    /// @brief Formatter is a manipulator.
    /// It allows you to format the message before writing it to the destination(s)
    ///
    /// Examples of formatters are : @ref formatter::time_t "prepend the time",
    /// @ref formatter::high_precision_time_t "prepend high-precision time",
    /// @ref formatter::idx_t "prepend the index of the message", etc.

    namespace formatter {

        /// @brief What to use as base class, for your formatter classes
        struct HPX_LOCAL_EXPORT manipulator
        {
            virtual void operator()(std::ostream&) const = 0;

            friend void format_value(std::ostream& os,
                boost::string_ref /*spec*/, manipulator const& value)
            {
                value(os);
            }

            /// @brief Override this if you want to allow configuration through
            /// scripting.
            ///
            /// That is, this allows configuration of your manipulator at run-time.
            virtual void configure(std::string const&) {}

            virtual ~manipulator();

        protected:
            // signify that we're only a base class - not to be used directly
            manipulator() = default;
        };

    }    // namespace formatter

    /// @brief Destination is a manipulator. It contains a place where the message,
    /// after being formatted, is to be written to.
    ///
    /// Some viable destinations are : @ref destination::cout "the console",
    /// @ref destination::file "a file", a socket, etc.
    namespace destination {

        /// @brief What to use as base class, for your destination classes
        struct HPX_LOCAL_EXPORT manipulator
        {
            virtual void operator()(message const&) = 0;

            /// @brief Override this if you want to allow configuration through
            /// scripting.
            ///
            /// That is, this allows configuration of your manipulator at run-time.
            virtual void configure(std::string const&) {}

            virtual ~manipulator();

        protected:
            // signify that we're only a base class - not to be used directly
            manipulator() = default;
        };

    }    // namespace destination

}}}    // namespace hpx::util::logging
