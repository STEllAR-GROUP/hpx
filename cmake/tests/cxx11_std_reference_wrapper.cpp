////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <functional>

int main()
{
    int lvalue = 0;
    std::reference_wrapper<int> wrapper = std::ref(lvalue);
}
