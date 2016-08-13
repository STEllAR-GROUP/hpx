//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates how a parallel::sort_by_key algorithm could be
// implemented based on the existing algorithm hpx::parallel::sort.

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_sort.hpp>
#include <hpx/include/iostreams.hpp>

#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void print_sequence(std::vector<int> const& keys, std::vector<char> const& values)
{
    for (std::size_t i = 0; i != keys.size(); ++i)
    {
        hpx::cout << "[" << keys[i] << ", " << values[i] << "]";
        if (i != keys.size() - 1)
            hpx::cout << ", ";
    }
    hpx::cout << std::endl;
}

int hpx_main()
{
    {
        std::vector<int> keys =
        {
            1,   4,   2,   8,   5,   7,   1,   4,   2,   8,   5,   7,
            1,   4,   2,   8,   5,   7,   1,   4,   2,   8,   5,   7,
            1,   4,   2,   8,   5,   7,   1,   4,   2,   8,   5,   7
        };
        std::vector<char> values =
        {
            'a', 'b', 'c', 'd', 'e', 'f', 'a', 'b', 'c', 'd', 'e', 'f',
            'a', 'b', 'c', 'd', 'e', 'f', 'a', 'b', 'c', 'd', 'e', 'f',
            'a', 'b', 'c', 'd', 'e', 'f', 'a', 'b', 'c', 'd', 'e', 'f'
        };

        hpx::cout << "unsorted sequence: {";
        print_sequence(keys, values);

        hpx::parallel::sort_by_key(
            hpx::parallel::par,
            keys.begin(),
            keys.end(),
            values.begin());

        hpx::cout << "sorted sequence:   {";
        print_sequence(keys, values);
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);   // Initialize and run HPX
}

