////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <random>

int main()
{
    // engines
    {
        std::random_device random_device;
        std::mt19937 mt19937;
        std::mt19937_64 mt19937_64;
    }

    // distributions
    {
        std::uniform_int_distribution<> uniform_int_distribution;
        std::uniform_real_distribution<> uniform_real_distribution;
    }
}
