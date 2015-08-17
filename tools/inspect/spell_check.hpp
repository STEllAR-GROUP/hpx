//  extra_whitespace_check header  ------------------------------------------//

//  Copyright (c) 2015 Brandon Cordes
//  Based on the apple_macro_check checker by Marshall Clow
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_SPELL_CHECK_HPP
#define BOOST_SPELL_CHECK_HPP

#include <fstream>
#include <memory>
#include "inspector.hpp"
#include "dictionary.hpp"

namespace boost
{
    namespace inspect
    {
        class spell_check : public inspector
        {
            long m_files_with_errors;
        public:

            spell_check();
            std::set<std::string> words;

            void open_file()
            {
                int wordcount = 0;
                string temp;
                for (std::size_t j = 0; j < sizeof(dictionary)/sizeof(dictionary[0]); j++)
                {
                    std::string temp;
                    for (std::size_t i = 0; dictionary[j][i] != 0; i++)
                        temp.push_back(tolower(dictionary[j][i]));
                    words.insert(temp);
                    wordcount++;
                }
                std::cout << wordcount << " words loaded\n";
            }

            std::string a = "*SEF*";
            std::string b = "Spelling error found";
            virtual const char * name() const { return a.c_str(); }
            virtual const char * desc() const { return b.c_str(); }

            virtual void inspect(
                const std::string & library_name,
                const path & full_path,
                const std::string & contents);

            virtual void print_summary(std::ostream& out)
            {
                string c = " files with spelling errors";
                out << "  " << m_files_with_errors << c << line_break();
            }

            virtual ~spell_check() {}
        };
    }
}

#endif // BOOST_EXTRA_WHITESPACE_CHECK_HPP
