//  extra_whitespace_check header  --------------------------------------------------------//

//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Based on the apple_macro_check checker by Marshall Clow and deprecated_macro_check by Eric Niebler
//  Based on the apple_macro_check checker by Marshall Clow
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_EXTRA_WHITESPACE_CHECK_HPP
#define BOOST_EXTRA_WHITESPACE_CHECK_HPP

#include "inspector.hpp"

namespace boost
{
    namespace inspect
    {
        class whitespace_check : public inspector
        {
            long m_files_with_errors;
        public:

            whitespace_check();
            virtual const char * name() const { return "*Endline Whitespace*"; }
            virtual const char * desc() const { return "Unnecessary whitespace at end of file"; }

            virtual void inspect(
                const std::string & library_name,
                const path & full_path,
                const std::string & contents);

            virtual void print_summary(std::ostream& out)
            {
                out << "  " << m_files_with_errors << " files with endline whitespace" << line_break();
            }

            virtual ~whitespace_check() {}
        };
    }
}

#endif // BOOST_EXTRA_WHITESPACE_CHECK_HPP
