//  Copyright 2011 Vicente J. Botet Escriba

//  Distributed under the Boost Software License, Version 1.0.
//  See http://www.boost.org/LICENSE_1_0.txt

#include <boost/chrono/chrono.hpp>
#include <boost/chrono/stopwatches/reporters/stopwatch_reporter.hpp>
#include <boost/chrono/stopwatches/simple_stopwatch.hpp>
#include <boost/chrono/process_cpu_clocks.hpp>
#include <vector>

#if 0
#include <sys/time.h> //for gettimeofday and timeval
#include <sys/times.h> //for times
#include <unistd.h>
#endif

//static const std::size_t size = 10000000;
  static const std::size_t size = 2000000;

typedef boost::chrono::simple_stopwatch<boost::chrono::high_resolution_clock> Stopwatch;
typedef boost::chrono::stopwatch_reporter<Stopwatch> Reporter;

template <typename Clock>
void perf_constant(std::vector<typename Clock::time_point>& vec)
{
  for (int i=size-1; i>=0; --i)
  {
    vec[i]=typename Clock::time_point();
  }
}

template <typename Clock>
void perf(std::vector<typename Clock::time_point>& vec)
{
  for (int i=size-1; i>=0; --i)
  {
    vec[i]=Clock::now();
  }
}

template <typename Clock>
void test()
{
  std::vector<typename Clock::time_point> vec(size);
  Stopwatch sw1;
  perf_constant<Clock>(vec);
  Stopwatch::duration t1 = sw1.elapsed();
  Stopwatch sw2;
  perf<Clock>(vec);
  Stopwatch::duration t2 = sw2.elapsed();
  std::cout <<" "<< (t2-t1)/size << std::endl;
  std::cout <<" "<< t2 << std::endl;
  std::cout <<" "<< t1 << std::endl;
  std::size_t cnt=0;
  for (int i=size-1; i>0; --i)
  {
    if (vec[i]!=vec[i-1]) ++cnt;
  }
  std::cout <<"changes: "<< cnt << std::endl;

}





#if 0
void perf2(std::vector<clock_t>& vec)
{
  Reporter rp;
  for (int i=size-1; i>=0; --i)
  {
    tms tm;
    vec[i]=::times(&tm);
  }
}

void perf3(std::vector<clock_t>& vec)
{
  Reporter rp;
  for (int i=size-1; i>=0; --i)
  {
    vec[i]=::clock();
  }
}

void test2()
{
  std::vector<clock_t> vec(size);
  perf2(vec);
  std::size_t cnt=0;
  for (int i=10; i>0; --i)
  {
    if (vec[i]!=vec[i-1]) ++cnt;
    std::cout << vec[i] << " " ;
  }
  std::cout<< std::endl;
  std::cout <<"changes: "<< cnt << std::endl;
}

void test3()
{
  std::vector<clock_t> vec(size);
  perf3(vec);
  std::size_t cnt=0;
  for (int i=10; i>0; --i)
  {
    if (vec[i]!=vec[i-1]) ++cnt;
    std::cout << vec[i] << " " ;
  }
  std::cout<< std::endl;
  std::cout <<"changes: "<< cnt << std::endl;
}

#endif

int main() {

  std::cout << "system_clock ";
  test<boost::chrono::system_clock>();
#ifdef BOOST_CHRONO_HAS_CLOCK_STEADY
  std::cout << "steady_clock " ;
  test<boost::chrono::steady_clock>();
#endif
  std::cout << "high_resolution_clock " ;
  test<boost::chrono::high_resolution_clock>();

#if defined(BOOST_CHRONO_HAS_PROCESS_CLOCKS)
  std::cout << "process_real_cpu_clock ";
  test<boost::chrono::process_real_cpu_clock>();
  std::cout << "process_user_cpu_clock ";
  test<boost::chrono::process_user_cpu_clock>();
  std::cout << "process_system_cpu_clock " ;
  test<boost::chrono::process_system_cpu_clock>();
  std::cout << "process_cpu_clock " ;
  test<boost::chrono::process_cpu_clock>();
#endif
  std::cout << "system_clock ";
  test<boost::chrono::system_clock>();
#if 0
  std::cout << "times ";
  test2();
  std::cout << "clock ";
  test3();
#endif
  return 1;
}
