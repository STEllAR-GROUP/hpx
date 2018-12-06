////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <regex>
#include <string>

int main()
{
    std::string text = "Lorem ipsum dolor sit amet";

    std::regex re("[a-z]+");
    std::smatch match;
    if (std::regex_match(text, match, re))
    {
        std::ssub_match smatch = match[0];
    }
}
