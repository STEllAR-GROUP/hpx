////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <unordered_set>

int main()
{
    std::unordered_set<unsigned> u;
    u.insert(0xFF0000);
    u.insert(0x00FF00);
    u.insert(0x0000FF);

    std::unordered_multiset<unsigned> um;
    um.insert(0xFF0000);
    um.insert(0x00FF00);
    um.insert(0x0000FF);
}
