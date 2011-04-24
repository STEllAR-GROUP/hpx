////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/cstdint.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/hardware/timestamp.hpp>

volatile int global = 0;

int main() {
    std::cout << "basic test" << std::endl;

    { // Basic test. 
        boost::uint64_t t0 = 0, t1 = 0;

        t0 = hpx::util::hardware::timestamp();

        for (unsigned i = 0; i < (1 << 16); ++i)
            ++global;

        t1 = hpx::util::hardware::timestamp();

        std::cout <<   "Tick 0: " << t0 
                  << "\nTick 1: " << t1 << std::endl;

        HPX_TEST(t1 > t0);
    }
   
    std::cout << "serializing test" << std::endl;
 
    { // Make sure that the timestamps are serialized.
        boost::uint64_t t0 = 0, t1 = 0;

        t0 = hpx::util::hardware::timestamp();
        t1 = hpx::util::hardware::timestamp();

        std::cout <<   "Tick 0: " << t0 
                  << "\nTick 1: " << t1 << std::endl;

        HPX_TEST(t1 > t0);
    }

    return hpx::util::report_errors();
}

