//  apple_macro_check header  --------------------------------------------------------//

//  Copyright Marshall Clow 2007.
//  Based on the tab-check checker by Beman Dawes
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "inspector.hpp"

namespace boost { namespace inspect {
    class apple_macro_check : public inspector
    {
        long m_files_with_errors;

    public:
        apple_macro_check();
        virtual const char* name() const
        {
            return "*APPLE-MACROS*";
        }
        virtual const char* desc() const
        {
            return "calls to Apple's debugging macros in file";
        }

        virtual void inspect(const std::string& library_name,
            const path& full_path, const std::string& contents);

        virtual void print_summary(std::ostream& out)
        {
            out << "  " << m_files_with_errors << " files with Apple macros"
                << line_break();
        }

        virtual ~apple_macro_check() {}
    };
}}    // namespace boost::inspect
