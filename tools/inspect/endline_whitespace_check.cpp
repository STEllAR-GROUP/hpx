﻿//  extra_whitespace_check implementation  ------------------------------------------------//

//  Copyright (c) 2015 Brandon Cordes
//  Based on the apple_macro_check checker by Marshall Clow and deprecated_macro_check by Eric Niebler
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "endline_whitespace_check.hpp"
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
        whitespace_check::whitespace_check()
            : m_files_with_errors(0)

        {
            register_signature(".c");
            register_signature(".cpp");
            register_signature(".cxx");
            register_signature(".h");
            register_signature(".hpp");
            register_signature(".hxx");
            register_signature(".ipp");
        }

        void whitespace_check::inspect(
            const string & library_name,
            const path & full_path,   // example: c:/foo/boost/filesystem/path.hpp
            const string & contents)     // contents of file to be inspected
        {
            if (contents.find("hpxinspect:" "noascii") != string::npos)
                return;

            string whitespace(" \t\f\v\r\n"), total, linenum;
            long errors = 0, currline = 0;
            size_t p = 0, extend = 0;
            vector<string> someline, lineorder;


            char_separator<char> sep("\n");
            tokenizer<char_separator<char>> tokens(contents, sep);
            for (const auto& t : tokens) {
                size_t rend = t.find_first_of("\r"), size = t.size();
                if (rend == size - 1)
                {
                    someline.push_back(t);
                }
                else
                {
                    char_separator<char> sep2("\r");
                    tokenizer<char_separator<char>> tokens2(t, sep2);
                    for (const auto& u : tokens2) {
                        someline.push_back(u);
                    }
                }
            }
            while (p < someline.size())
            {
                currline++;
                size_t rend = someline[p].find_last_of("\r"), found = someline[p].find_last_not_of(whitespace);
                if (rend != string::npos)
                {
                    extend = 2;
                }
                else
                {
                    extend = 1;
                }
                size_t size = someline[p].size();
                if (found < size - extend)
                {
                    errors++;
                    linenum = to_string(currline);
                    lineorder.push_back(linenum);
                }
                p++;
            }
            p = 0;
            while (p < lineorder.size())
            {
                total += lineorder[p];
                if (p < lineorder.size() - 1)
                {
                    total += ", ";
                }
                p++;
            }
            // Make the 'find' function search forward from the next character onwards
            if (errors > 0)
            {
                error(library_name, full_path, string("*Endline Whitespace*: ") + total);
                ++m_files_with_errors;
            }
        }
    } // namespace inspect
} // namespace boost
