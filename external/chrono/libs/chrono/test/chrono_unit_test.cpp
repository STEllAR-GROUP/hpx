//  chrono_unit_test.cpp  ----------------------------------------------------//

//  Copyright 2008 Beman Dawes

//  Distributed under the Boost Software License, Version 1.0.
//  See http://www.boost.org/LICENSE_1_0.txt

#include <boost/chrono.hpp>
#include <iostream>


int main()
{
  boost::chrono::nanoseconds nanosecs;
  boost::chrono::microseconds microsecs;
  boost::chrono::milliseconds millisecs;
  boost::chrono::seconds secs;
  boost::chrono::minutes mins;
  boost::chrono::hours hrs;

  std::time_t sys_time
    = boost::chrono::system_clock::to_time_t(boost::chrono::system_clock::now());

  std::cout
    << "system_clock::to_time_t(system_clock::now()) is "
    << std::asctime(std::gmtime(&sys_time)) << std::endl;

  return 0;
}
