//  character_length_check implementation  ----------------------------------//

//  Copyright (c) 2015 Brandon Cordes
//  Based on the apple_macro_check checker by Marshall Clow
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "length_check.hpp"
#include "function_hyper.hpp"
#include <iostream>
#include <functional>
#include <string>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include "boost/regex.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/filesystem/operations.hpp"

using namespace std;
namespace fs = boost::filesystem;

namespace boost
{
    namespace inspect
    {
        size_t limit;

        length_check::length_check(size_t setting)
            : m_files_with_errors(0)

        {
            register_signature(".c");
            register_signature(".cpp");
            register_signature(".css");
            register_signature(".cxx");
            register_signature(".h");
            register_signature(".hpp");
            register_signature(".hxx");
            register_signature(".inc");
            register_signature(".ipp");
            register_signature(".txt");

            limit = setting;
        }

        struct pattern
        {
            char const* const rx_;
            int pos_;
        };

        static pattern const patterns[] =
        {
            { "\\s*#\\s*error", 0 },
            { "\\s*#\\s*include", 0 },
            { "https?://", -1 },
        };

        void length_check::inspect(
            const string & library_name,
            const path & full_path,   // ex: c:/foo/boost/filesystem/path.hpp
            const string & contents)     // contents of file to be inspected
        {
            if (contents.find("hpxinspect:" "linelength") != string::npos)
                return;

            string pathname = full_path.string();
            if (pathname.find("CMakeLists.txt") != string::npos)
                return;

            // Temporary, until we are ready to format documentation files in
            // this limitation.
            if (library_name.find(".qbk") != string::npos)
                return;

            string total, linenum;
            long errors = 0, currline = 0;
            size_t p = 0;
            vector<string> someline, lineorder;

            char_separator<char> sep("\n", "", boost::keep_empty_tokens);
            tokenizer<char_separator<char>> tokens(contents, sep);
            for (const auto& t : tokens) {
                size_t rend = t.find_first_of("\r"), size = t.size();
                if (rend == size - 1)
                {
                    someline.push_back(t);
                }
                else
                {
                    char_separator<char> sep2("\r", "", boost::keep_empty_tokens);
                    tokenizer<char_separator<char>> tokens2(t, sep2);
                    for (const auto& u : tokens2) {
                        someline.push_back(u);
                    }
                }
            }

            while (p < someline.size())
            {
                currline++;
                size_t rend = someline[p].find_last_of("\r");
                bool check_not = 0;
                boost::regex error_note, http_note;

                for (std::size_t i = 0;
                     i != sizeof(patterns)/sizeof(patterns[0]);
                     ++i)
                {
                    boost::regex rx(patterns[i].rx_);
                    boost::smatch m;
                    if (boost::regex_search(someline[p], m, rx))
                    {
                        // skip this line if either no position is specified
                        // or the pattern was found at the given position
                        if (patterns[i].pos_ == -1 || m.position() == patterns[i].pos_)
                            check_not = 1;
                    }
                }

                size_t size = someline[p].size();
                if (size > limit && check_not == 0)
                {
                    errors++;
                    linenum = to_string(currline);
                    lineorder.push_back(linenum);
                }
                ++p;
            }

            p = 0;
            while (p < lineorder.size())
            {
                total += linelink(full_path, lineorder[p]);
                //linelink is located in function_hyper.hpp
                if (p < lineorder.size() - 1)
                {
                    total += ", ";
                }
                p++;
            }
            if (errors > 0)
            {
                string errored = "Line length limit*: " + total;
                error(library_name, full_path, errored);
                ++m_files_with_errors;
            }
        }
    } // namespace inspect
} // namespace boost
