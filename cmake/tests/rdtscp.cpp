////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/cstdint.hpp>

#include <hpx/util/lightweight_test.hpp>

namespace {

inline boost::uint64_t tick() {
  boost::uint32_t lo(0), hi(0);
  __asm__ __volatile__ (
      "rdtscp\n"
    : "=a" (lo), "=d" (hi)
    :
    : "%ecx"
  );
  return ((static_cast<boost::uint64_t>(hi)) << 32) | lo;
}

volatile unsigned global = 0;

}

int main()
{
    { // Basic test. 
        boost::uint64_t t0 = 0, t1 = 0;

        t0 = tick();

        for (volatile unsigned i = 0; i < (1 << 16); ++i)
            ++global;

        t1 = tick();

        HPX_TEST(t1 > t0);
    }
   
    { // Make sure that the ticks are serialized.
        boost::uint64_t t0 = 0, t1 = 0;

        t0 = tick();
        t1 = tick();

        HPX_TEST(t1 > t0);
    }

    return hpx::util::report_errors();
}

