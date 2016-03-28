////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <string>

int main()
{
    std::string s1 = std::to_string(42);
    std::string s2 = std::to_string(42l);
    std::string s3 = std::to_string(42.f);
    std::string s4 = std::to_string(42.0);
}
