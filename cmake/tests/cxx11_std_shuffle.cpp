////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <random>
#include <vector>

int main()
{
    std::vector<int> v = {1, 2, 3};
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);

    return 0;
}
