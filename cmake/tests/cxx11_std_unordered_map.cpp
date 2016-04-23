////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <unordered_map>
#include <string>

int main()
{
    std::unordered_map<std::string, unsigned> u;
    u.insert(std::make_pair("RED", 0xFF0000));
    u.insert(std::make_pair("GREEN", 0x00FF00));
    u.insert(std::make_pair("BLUE", 0x0000FF));

    std::unordered_multimap<std::string, unsigned> um;
    um.insert(std::make_pair("RED", 0xFF0000));
    um.insert(std::make_pair("GREEN", 0x00FF00));
    um.insert(std::make_pair("BLUE", 0x0000FF));
}
