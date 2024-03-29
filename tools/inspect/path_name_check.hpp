//  long_name_check header  --------------------------------------------------//
//  (main class renamed to: file_name_check) - gps

//  Copyright Beman Dawes 2002.
//  Copyright Gennaro Prota 2006.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "inspector.hpp"

namespace boost { namespace inspect {
    class file_name_check : public inspector
    {
        long m_name_errors;

    public:
        file_name_check();

        virtual const char* name() const
        {
            return "*N*";
        }
        virtual const char* desc() const
        {
            return "file and directory name issues";
        }

        virtual void inspect(const string& library_name, const path& full_path);

        virtual void inspect(const string&,    // "filesystem"
            const path&,    // "c:/foo/boost/filesystem/path.hpp"
            const string&)
        { /* empty */
        }

        virtual void print_summary(std::ostream& out)
        {
            out << "  " << m_name_errors << " " << desc() << line_break();
        }

        virtual ~file_name_check() {}
    };
}}    // namespace boost::inspect
