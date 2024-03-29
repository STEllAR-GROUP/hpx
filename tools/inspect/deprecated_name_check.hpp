//  deprecated_name_check header  -------------------------------------------//

//  Copyright Beman Dawes   2002
//  Copyright Rene Rivera   2004.
//  Copyright Gennaro Prota 2006.
//  Copyright Hartmut Kaiser 2016.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "inspector.hpp"

#include "boost/regex.hpp"

#include <vector>

namespace boost { namespace inspect {
    struct deprecated_names
    {
        char const* name_regex;
        char const* use_instead;
    };

    struct deprecated_names_regex_data
    {
        deprecated_names_regex_data(
            deprecated_names const* d, std::string const& rx)
          : data(d)
          , pattern(rx, boost::regex::normal)
        {
        }

        deprecated_names const* data;
        boost::regex pattern;
    };

    class deprecated_name_check : public inspector
    {
        long m_errors;
        std::vector<deprecated_names_regex_data> regex_data;

    public:
        deprecated_name_check();
        virtual const char* name() const
        {
            return "*DN*";
        }
        virtual const char* desc() const
        {
            return "uses of deprecated names";
        }

        virtual void inspect(const std::string& library_name,
            const path& full_path, const std::string& contents);

        virtual void print_summary(std::ostream& out)
        {
            out << "  " << m_errors << " deprecated names" << line_break();
        }

        virtual ~deprecated_name_check() {}
    };
}}    // namespace boost::inspect
