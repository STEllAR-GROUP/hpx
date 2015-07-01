//  spell_check implementation  ----------------------------------//

//  Copyright (c) 2015 Brandon Cordes
//  Based on the apple_macro_check checker by Marshall Clow
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "spell_check.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <functional>
#include <string>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include "boost/regex.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/filesystem/operations.hpp"

namespace fs = boost::filesystem;

bool check_word(const string& str)
{
    for (std::size_t i = 0; i < str.size(); ++i)
        if (!((str[i] >= 'a' && str[i] <= 'z') ||
            (str[i] >= 'A' && str[i] <= 'Z') ||
            str[i] == ' '))
        return false;
    return true;
}

namespace
{
    //The file endings that you wish the whole file to be spell checked are here
    const char * file_ext[] = {
         ".qbk",
         NULL
    };

    boost::regex doxygen(
        "("
        "///[^\\n]*"           // single line comments (//)
        "|"
        "/\\*\\*.*?\\*/"         // multi line comments (/**/)
        ")"
        , boost::regex::normal);

    //(\/\/\/.*)|\/\*(?s).*?\*\/
} // unnamed namespace

namespace boost
{
    namespace inspect
    {
        spell_check::spell_check()
            : m_files_with_errors(0)
        {
            register_signature(".c");
            register_signature(".cpp");
            register_signature(".h");
            register_signature(".hpp");
            register_signature(".qbk");
        }

        void spell_check::inspect(
            const string & library_name,
            const path & full_path,   // ex: c:/foo/boost/filesystem/path.hpp
            const string & contents)     // contents of file to be inspected
        {
            if (contents.find("hpxinspect:" "spellerror") != string::npos)
                return;

            string total, path_name = full_path.string(), letters = "a";
            const char **ptr;
            bool is_file = false, is_correct = false;
            long errors = 0;
            size_t p = 0, q = 0;
            std::set<std::string>::iterator dictionary_place;
            std::vector<std::string> checking;
            std::vector<int> linenumb, wordline;
            //Loading all of the words currently in the dictionary
            //Change to the dictionary text file one day
            //This for loop determines if the file extention is one that requires the whole file to be read
            for (ptr = file_ext; *ptr != NULL; ++ptr)
            {
                size_t a = path_name.find(*ptr);
                if (a != string::npos) {
                    is_file = true;
                }
            }
            if (is_file)
            {
                int currline = 0;
                char_separator<char> sep("\n");
                tokenizer<char_separator<char> > tokens(contents, sep);
                for (const auto& t : tokens) {
                    checking.push_back(t);
                    currline++;
                    linenumb.push_back(currline);
                }
            }
            else
            {
                //This checks for doxygen comments
                boost::sregex_iterator cur(contents.begin(), contents.end(), doxygen), end;

                string words, total;

                for (; cur != end; ++cur /*, ++m_files_with_errors*/)
                {

                    if (!(*cur)[3].matched)
                    {
                        string::const_iterator it = contents.begin();
                        string::const_iterator match_it = (*cur)[0].first;

                        string::const_iterator line_start = it;

                        string::size_type line_number = 1;
                        for (; it != match_it; ++it) {
                            if (string::traits_type::eq(*it, '\n')) {
                                ++line_number;
                                line_start = it + 1; // could be end()
                            }
                        }
                        checking.push_back(std::string((*cur)[0].first, (*cur)[0].second));
                        linenumb.push_back(line_number);
                    }
                }
            }
            //This tokenizes based on space
            std::vector<std::string> strs;
            while (p < checking.size())
            {
                char_separator<char> sep(" \n\t\v");
                tokenizer<char_separator<char> > tokens(checking[p], sep);
                for (const auto& t : tokens)
                {
                     string hold = t, finald;
                     string::size_type begin, end;
                      transform(hold.begin(), hold.end(), hold.begin(), ::tolower);
                     begin = 0;
                     end = hold.find_last_of("abcdefghijklmnopqrstuvwxyz") + 1;
                     //Will use the size to change finald into only the word itself.
                     if (end < hold.npos)
                     {
                         finald = hold.substr(begin, end - begin);
                         bool is_word;
                         is_word = check_word(finald);
                         if (is_word)
                         {
                             if (finald.size() > 1)
                             {
                                 strs.push_back(finald);
                                 wordline.push_back(linenumb[p]);
                             }
                         }
                     }
                }
                p++;
            }
            p = 0;
            while (p < strs.size())
            {
                dictionary_place = words.find(strs[p]);
                if (dictionary_place != words.end())
                {
                    is_correct = true;
                }
                if (!is_correct)
                {
                    errors++;
                    total +=  "\"" + strs[p] + "\", in line " + std::to_string(wordline[p]) + ", ";
                }
                else
                {
                    is_correct = false;
                }
                q = 0;
                p++;
            }
            if (errors > 0)
            {
                total += "is not found in the dictionary.";
                string errored = "SEF*: " + total;
                error(library_name, full_path, errored);
                ++m_files_with_errors;
            }
        }
    } // namespace inspect
} // namespace boost
