//  Copyright (c) 2012 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <omp.h>

int main()
{
    #ifdef _OPENMP
        return 0;
    #else
        #error OpenMP support not found.
    #endif
}

