//  extra_whitespace_check implementation  ----------------------------------//

//  Copyright (c) 2015 Brandon Cordes
//  Based on the apple_macro_check checker by Marshall Clow
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/modules/string_util.hpp>

#include <functional>
#include <iostream>
#include <string>
#include "boost/regex.hpp"
#include "endline_whitespace_check.hpp"
#include "function_hyper.hpp"

using namespace std;
namespace fs = hpx::filesystem;

namespace boost { namespace inspect {
    whitespace_check::whitespace_check()
      : m_files_with_errors(0)

    {
        register_signature(".c");
        register_signature(".cpp");
        register_signature(".cu");
        register_signature(".css");
        register_signature(".cxx");
        register_signature(".h");
        register_signature(".hpp");
        register_signature(".hxx");
        register_signature(".inc");
        register_signature(".ipp");
        register_signature(".txt");
        register_signature(".ixx");
    }

    void whitespace_check::inspect(const string& library_name,
        const path& full_path,     // ex: c:/foo/boost/filesystem/path.hpp
        const string& contents)    // contents of file to be inspected
    {
        if (contents.find("hpxinspect:"
                          "endlinewhitespace") != string::npos)
            return;

        string whitespace(" \t\f\v\r\n"), total, linenum;
        long errors = 0, currline = 0;
        size_t p = 0, extend = 0;
        vector<string> someline, lineorder;

        hpx::string_util::char_separator sep(
            "\n", "", hpx::string_util::empty_token_policy::keep);
        hpx::string_util::tokenizer tokens(contents, sep);
        for (const auto& t : tokens)
        {
            size_t rend = t.find_first_of("\r"), size = t.size();
            if (rend == size - 1)
            {
                someline.push_back(t.substr(0, t.size() - 1));
            }
            else
            {
                hpx::string_util::char_separator sep2(
                    "\r", "", hpx::string_util::empty_token_policy::keep);
                hpx::string_util::tokenizer tokens2(t, sep2);
                for (const auto& u : tokens2)
                {
                    if (!u.empty() && u.back() == '\r')
                        someline.push_back(u.substr(0, u.size() - 1));
                    else
                        someline.push_back(u);
                }
            }
        }
        while (p < someline.size())
        {
            currline++;
            size_t rend = someline[p].find_last_of("\r");
            size_t found = someline[p].find_last_not_of(whitespace);
            if (rend != string::npos)
            {
                extend = 2;
            }
            else
            {
                extend = 1;
            }
            size_t size = someline[p].size();
            if (found < size - extend ||
                (found == someline[p].npos && size > 1))
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
            string errored = "*Endline Whitespace*: " + total;
            error(library_name, full_path, errored);
            ++m_files_with_errors;
        }
    }
}}    // namespace boost::inspect
