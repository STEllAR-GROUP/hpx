//  Copyright 2011 Vicente J. Botet Escriba

//  Distributed under the Boost Software License, Version 1.0.
//  See http://www.boost.org/LICENSE_1_0.txt

#include <boost/chrono/chrono.hpp>
#include <boost/chrono/chrono_io.hpp>

  static
  const int
  is_leap_table_[400] =
  {
      1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,
      0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,
      0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,
      0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0
  };
  /**
   * is_leap could be made more efficient by using a table indexed by the y % 400.
   * This table could contain true for y such as
   * y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
   *
   * is_leap = is_leap_table[y%400]
   */
  static
  inline
  bool
  is_leap_calc(int y) BOOST_NOEXCEPT
  {
    return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
  }


  static
  inline
  bool
  is_leap_table(int y) BOOST_NOEXCEPT
  {
    return (y % 4 == 0) && is_leap_table_[y%400];
    //return (y & 3 == 0) && is_leap_table_[y%400];
  }

  static
  inline
  bool
  is_leap(int y) BOOST_NOEXCEPT
  {
#if defined BOOST_CHRONO_IS_LEAP_USES_TABLE
    return is_leap_table(y);
#else
    return is_leap_calc(y);
#endif
  }

int main()
{
    typedef boost::chrono::high_resolution_clock Clock;
    typedef boost::chrono::duration<double> sec;
//    for (int y = -32768; y <= 32767; ++y)
//    {
//      if (is_leap_calc(y)!=is_leap_table(y))
//        std::cout
//        << "is_leap_calc:" << is_leap_calc(y) << " "
//        << "is_leap_table:" << is_leap_table(y) << " "
//        << y << " " << y%400 <<"\n";
//
//    }
    {
      int cnt=0;
    Clock::time_point t0 = Clock::now();
    for (int x = 0; x < 100; ++x)
    for (int y = 0; y <= 32767; ++y)
    //for (int y = -32768; y <= 32767; ++y)
    {
      cnt += is_leap_calc(y)?1:0;
    }
    Clock::time_point t1 = Clock::now();
    std::cout << "is_leap_calc: " << t1-t0 << " " << cnt <<"\n";
    }

    {
      int cnt=0;
    Clock::time_point t0 = Clock::now();
    for (int x = 0; x < 100; ++x)
      for (int y = 0; y <= 32767; ++y)
      //for (int y = -32768; y <= 32767; ++y)
    {
      cnt += is_leap_table(y)?1:0;
    }
    Clock::time_point t1 = Clock::now();
    std::cout << "is_leap_table:" << t1-t0 << " " << cnt <<"\n";
    }





    return 0;


}
