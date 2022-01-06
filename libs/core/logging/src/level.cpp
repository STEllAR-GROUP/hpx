//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/logging/level.hpp>

#if defined(HPX_HAVE_LOGGING)
#include <boost/utility/string_ref.hpp>

#include <cstddef>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace logging {

    static std::string levelname(level value)
    {
        switch (value)
        {
        case hpx::util::logging::level::enable_all:
            return "<all>";
        case hpx::util::logging::level::debug:
            return "<debug>";
        case hpx::util::logging::level::info:
            return "<info>";
        case hpx::util::logging::level::warning:
            return "<warning>";
        case hpx::util::logging::level::error:
            return "<error>";
        case hpx::util::logging::level::fatal:
            return "<fatal>";
        case hpx::util::logging::level::always:
            return "<always>";
        default:
            break;
        }

        return '<' + std::to_string(static_cast<int>(value)) + '>';
    }

    void format_value(std::ostream& os, boost::string_ref spec, level value)
    {
        if (!spec.empty())
            throw std::runtime_error("Not a valid format specifier");

        os << std::right << std::setfill(' ') << std::setw(10)
           << levelname(value);
    }

}}}    // namespace hpx::util::logging

#endif    // HPX_HAVE_LOGGING
