////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Beman Dawes
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

int main()
{
    // example from 6.5.4 The range-based for statement [stmt.ranged]
    int array[5] = { 1, 2, 3, 4, 5 };
    for (int& x : array)
    x *= 2;
}
