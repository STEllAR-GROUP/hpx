//  deprecated macro check implementation  ----------------------------------//
//  Protect against ourself: hpxinspect:nodeprecated_macros

//  Copyright (c) 2015 Brandon Cordes
//  Based on the depreciated_macro_check by Eric Niebler
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include "preferred_namespace_check.hpp"
#include <functional>
#include <vector>
#include <array>
#include <string>
#include <boost/regex.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include "boost/lexical_cast.hpp"
#include "boost/filesystem/operations.hpp"

namespace fs = boost::filesystem;

namespace
{
    //Add additional unwanted namespaces followed by the preferred namespace
    struct floo
    {
        const char const * a;
        const char const * b;
        const char const * c;
    };

    floo const named[] = {
        {"boost\\s*::\\s*move\\s*", "boost::move", "std::move"},
        {"std\\s*::\\s*begin\\s*", "std::begin", "boost::begin"},
        {"std\\s*::\\s*end\\s*[^l]","std::end", "boost::end"},
        {"include\\s*<regex>\\s*","include <regex>", "include <boost/regex.hpp>"},
        {NULL, NULL, NULL}
    };
} // unnamed namespace

namespace boost
{
    namespace inspect
    {
        preferred_namespace_check::preferred_namespace_check()
            : m_files_with_errors(0)
            , m_from_boost_root(
                fs::exists(search_root_path() / "boost") &&
                fs::exists(search_root_path() / "libs"))
        {
            register_signature(".c");
            register_signature(".cpp");
            register_signature(".cxx");
            register_signature(".h");
            register_signature(".hpp");
            register_signature(".hxx");
            register_signature(".ipp");
        }

        void preferred_namespace_check::inspect(
            const string & library_name,
            const path & full_path,   // ex: c:/foo/boost/filesystem/path.hpp
            const string & contents)     // contents of file to be inspected
        {
            if (contents.find("boostinspect:" "npref_namespace") != string::npos)
                return;
            //Place characters needed to be detected but not written here
            std::vector<boost::regex> check;
            std::vector<std::string> replace_with, preferred, file;
            std::vector<long> linenumb;
            std::size_t p = 0, q = 0;
            const floo *ptr;
            long errors = 0, currline = 0;
            //For line number
            char_separator<char> sep("\n", "", boost::keep_empty_tokens);
            tokenizer<char_separator<char>> tokens(contents, sep);
            for (const auto& t : tokens) {
                size_t rend = t.find_first_of("\r"), size = t.size();
                if (rend == size - 1)
                {
                    file.push_back(t);
                    currline++;
                    linenumb.push_back(currline);

                }
                else
                {
                    char_separator<char> sep2("\r", "", boost::keep_empty_tokens);
                    tokenizer<char_separator<char>> tokens2(t, sep2);
                    for (const auto& u : tokens2) {
                        file.push_back(u);
                        currline++;
                        linenumb.push_back(currline);
                    }
                }
            }
            for (ptr = named; ptr->a != NULL; ++ptr)
            {
                std::string b, c;
                boost::regex a;
                b = ptr->b;
                replace_with.push_back(b);
                a = ptr->a;
                check.push_back(a);
                b = ptr->c;
                preferred.push_back(b);
            }
            while (p < check.size())
            {
                while (q < file.size())
                {
                    if (regex_search(file[q], check[p])) {
                        ++errors;
                        std::string replaced = "Replace \"" + replace_with[p];
                        std::string with = replaced + "\" in line ";
                        std::string the = with + std::to_string(linenumb[q]) + " with \"";
                        std::string correct = the + preferred[p] + "\"";
                        error(library_name, full_path, correct);
                    }
                    q++;
                }
                q = 0;
                p++;
            }

            if (errors > 0)
                ++m_files_with_errors;
        }
    } // namespace inspect
} // namespace boost
