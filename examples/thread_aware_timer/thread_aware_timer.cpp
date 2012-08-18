//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Including 'hpx/hpx_main.hpp' instead of the usual 'hpx/hpx_init.hpp' enables 
// to use the plain C-main below as the direct main HPX entry point.
#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/util.hpp>

#define NUMTESTS 10000

int main(int argc, char* argv[])
{
    hpx::util::thread_aware_timer tat;

    double elapsed = 0;
    for (int i = 0; i < NUMTESTS; ++i) {
        hpx::util::high_resolution_timer t;
        tat.elapsed();
        elapsed += t.elapsed();
    }

    hpx::cout << "Average time required to query the thread aware timer: "
              << elapsed/NUMTESTS << "\n" << hpx::flush;
    return 0;
}

