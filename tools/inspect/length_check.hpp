//  character_length_check header  ------------------------------------------//

//  Copyright (c) 2015 Brandon Cordes
//  Based on the apple_macro_check checker by Marshall Clow
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_EIGHTY_CHECK_HPP
#define BOOST_EIGHTY_CHECK_HPP

#include "inspector.hpp"

namespace boost
{
    namespace inspect
    {
        class length_check : public inspector
        {
            long m_files_with_errors;
        public:

            length_check(size_t setting);

            std::string a = "*Line length limit*";
            std::string b = "The line is longer than allowed by the character limit";
            virtual const char * name() const { return a.c_str(); }
            virtual const char * desc() const { return b.c_str(); }

            virtual void inspect(
                const std::string & library_name,
                const path & full_path,
                const std::string & contents);

            virtual void print_summary(std::ostream& out)
            {
                string c = " lines exceeding the character limit";
                out << "  " << m_files_with_errors << c << line_break();
            }

            virtual ~length_check() {}
        };
    }
}

#endif // BOOST_EXTRA_WHITESPACE_CHECK_HPP
