////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/cstdint.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/hardware/tick.hpp>

int main() {
  boost::uint64_t t0 = 0, t1 = 0;

  t0 = hpx::util::hardware::tick();
  std::cout << "Tick 0: " << t0 << std::endl;

  t1 = hpx::util::hardware::tick();
  std::cout << "Tick 1: " << t1 << std::endl;

  BOOST_TEST(t1 > t0);
}

