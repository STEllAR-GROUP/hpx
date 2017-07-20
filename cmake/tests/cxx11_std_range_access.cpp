////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iterator>
#include <vector>

int main()
{
    int array[3] = { 0, 1, 2 };
    int* array_begin = std::begin(array);
    int* array_end = std::begin(array);

    std::vector<int> vector(3);
    std::vector<int>::iterator vector_begin = std::begin(vector);
    std::vector<int>::iterator vector_end = std::begin(vector);
};
