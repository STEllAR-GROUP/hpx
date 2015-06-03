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
#include "boost/regex.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/filesystem/operations.hpp"

namespace fs = boost::filesystem;

namespace
{
    //Add additional unwanted namespaces in dont_want
    const char * dont_want[] = {
        "boost::move(",
        "using boost::move;",
        "std::begin(",
        "using std::begin;",
        "std::end(",
        "using std::end;",
        NULL
    };
    //Add replacements for each dont_want in want
    //both dont_want and want must be the same size
    const char * want[] = {
        "std::move",
        "using std::move",
        "boost::begin",
        "using boost::begin",
        "boost::end",
        "using boost::end",
        NULL
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
            std::string junk(" ();");
            std::vector<std::string> not_preferred, preferred;
            std::size_t p = 0, foundn, foundp;
            const char **ptr;
            long errors = 0;
            for (ptr = dont_want; *ptr != NULL; ++ptr)
            {
                not_preferred.push_back(*ptr);
            }
            for (ptr = want; *ptr != NULL; ++ptr)
            {
                preferred.push_back(*ptr);
            }
            while (p < not_preferred.size() && p < preferred.size())
            {
                if (contents.find(not_preferred[p]) != string::npos) {
                    foundn = not_preferred[p].find_last_not_of(junk);
                    if (foundn != not_preferred[p].size())
                        not_preferred[p].erase(foundn + 1);
                    foundp = not_preferred[p].find_last_not_of(junk);
                    if (foundp != not_preferred[p].size())
                        not_preferred[p].erase(foundp + 1);
                    ++errors;
                    std::string replace = "Replace \"" + not_preferred[p];
                    std::string with = replace + "\" with \"";
                    std::string correct = with + preferred[p] + "\"";
                    error(library_name, full_path, correct);
                }
                p++;
            }

            if (errors > 0)
                ++m_files_with_errors;
        }
    } // namespace inspect
} // namespace boost
