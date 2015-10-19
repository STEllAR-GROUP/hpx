//  extra_whitespace_check header  ------------------------------------------//

//  Copyright (c) 2015 Brandon Cordes
//  Based on the apple_macro_check checker by Marshall Clow
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PREFERRED_NAMESPACE_CHECK_HPP
#define BOOST_PREFERRED_NAMESPACE_CHECK_HPP

#include "inspector.hpp"


namespace boost
{
    namespace inspect
    {
        class preferred_namespace_check : public inspector
        {
            long m_files_with_errors;
            bool m_from_boost_root;
        public:

            preferred_namespace_check();
            std::string a = "*USE INSTEAD*";
            std::string b = "HPX prefers the suggested namespace over the current one.";
            virtual const char * name() const { return a.c_str(); }
            virtual const char * desc() const { return b.c_str(); }

            virtual void inspect(
                const std::string & library_name,
                const path & full_path,
                const std::string & contents);

            virtual void print_summary(std::ostream& out)
            {
                std::string c = " files with incorrect namespaces";
                out << "  " << m_files_with_errors << c << line_break();
            }

            virtual ~preferred_namespace_check() {}
        };
    }
}

#endif // BOOST_DEPRECATED_MACRO_CHECK_HPP
