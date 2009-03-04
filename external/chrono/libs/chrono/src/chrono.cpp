//  chrono.cpp  --------------------------------------------------------------//

//  Copyright Beman Dawes 2008

//  Distributed under the Boost Software License, Version 1.0.
//  See http://www.boost.org/LICENSE_1_0.txt

// define BOOST_CHRONO_SOURCE so that <boost/filesystem/config.hpp> knows
// the library is being built (possibly exporting rather than importing code)
#define BOOST_CHRONO_SOURCE 

#include <boost/chrono/chrono.hpp>
#include <boost/system/system_error.hpp>
#include <boost/throw_exception.hpp>

//----------------------------------------------------------------------------//
//                                                                            //
//                     Platform-specific Implementations                      //
//                                                                            //
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//                                Windows                                     //
//----------------------------------------------------------------------------//
#if defined(BOOST_CHRONO_WINDOWS_API)

#include <windows.h>
#undef min
#undef max

namespace
{
  double get_nanosecs_per_tic()
  {
    LARGE_INTEGER freq;
    if ( !QueryPerformanceFrequency( &freq ) )
      return 0.0L;
    return 1000000000.0L / freq.QuadPart;
  }

  const double nanosecs_per_tic = get_nanosecs_per_tic();
}

namespace boost
{
namespace chrono
{

  monotonic_clock::time_point monotonic_clock::now()
  {

    LARGE_INTEGER pcount;
    if ( nanosecs_per_tic <= 0.0L || !QueryPerformanceCounter( &pcount ) )
    {
      DWORD cause = (nanosecs_per_tic <= 0.0L ? ERROR_NOT_SUPPORTED : ::GetLastError());
      boost::throw_exception(
        system::system_error( cause, system::system_category, "chrono::monotonic_clock" ));
    }

    return time_point(duration(
      static_cast<monotonic_clock::rep>(nanosecs_per_tic * pcount.QuadPart) ));
  }

  monotonic_clock::time_point monotonic_clock::now( system::error_code & ec )
  {
    static double nanosecs_per_tic = get_nanosecs_per_tic();

    LARGE_INTEGER pcount;
    if ( nanosecs_per_tic <= 0.0L || !QueryPerformanceCounter( &pcount ) )
    {
      DWORD cause = (nanosecs_per_tic <= 0.0L ? ERROR_NOT_SUPPORTED : ::GetLastError());
      ec.assign( cause, system::system_category );
      return time_point(duration(0));
    }

    ec.clear();
    return time_point(duration(
      static_cast<monotonic_clock::rep>(nanosecs_per_tic * pcount.QuadPart) ));
  }

  system_clock::time_point system_clock::now()
  {
    FILETIME ft;
    ::GetSystemTimeAsFileTime( &ft );  // never fails
    return time_point(duration(
      (static_cast<__int64>( ft.dwHighDateTime ) << 32) | ft.dwLowDateTime));
  }

  system_clock::time_point system_clock::now( system::error_code & ec )
  {
    FILETIME ft;
    ::GetSystemTimeAsFileTime( &ft );  // never fails
    ec.clear();
    return time_point(duration(
      (static_cast<__int64>( ft.dwHighDateTime ) << 32) | ft.dwLowDateTime));
  }

  std::time_t system_clock::to_time_t(const system_clock::time_point& t)
  {
      __int64 temp = t.time_since_epoch().count();

  #   if !defined( BOOST_MSVC ) || BOOST_MSVC > 1300 // > VC++ 7.0
      temp -= 116444736000000000LL;  // delta from epoch in microseconds
  #   else
      temp -= 116444736000000000;
  #   endif

      temp /= 10000000;
      return static_cast<std::time_t>( temp );
  }

  system_clock::time_point system_clock::from_time_t(std::time_t t)
  {
      __int64 temp = t;
      temp *= 10000000;

  #   if !defined( BOOST_MSVC ) || BOOST_MSVC > 1300 // > VC++ 7.0
      temp += 116444736000000000LL;
  #   else
      temp += 116444736000000000;
  #   endif

      return time_point(duration(temp));
  }

}  // namespace chrono
}  // namespace boost

//----------------------------------------------------------------------------//
//                                 Mac                                        //
//----------------------------------------------------------------------------//
#elif defined(BOOST_CHRONO_MAC_API)

#include <sys/time.h> //for gettimeofday and timeval
#include <mach/mach_time.h>  // mach_absolute_time, mach_timebase_info_data_t

namespace boost
{
namespace chrono
{

// system_clock

// gettimeofday is the most precise "system time" available on this platform.
// It returns the number of microseconds since New Years 1970 in a struct called timeval
// which has a field for seconds and a field for microseconds.
//    Fill in the timeval and then convert that to the time_point
system_clock::time_point
system_clock::now()
{
    timeval tv;
    gettimeofday(&tv, 0);
    return time_point(seconds(tv.tv_sec) + microseconds(tv.tv_usec));
}

system_clock::time_point
system_clock::now(system::error_code & ec)
{
    timeval tv;
    gettimeofday(&tv, 0);
    ec.clear();
    return time_point(seconds(tv.tv_sec) + microseconds(tv.tv_usec));
}

// Take advantage of the fact that on this platform time_t is nothing but
//    an integral count of seconds since New Years 1970 (same epoch as timeval).
//    Just get the duration out of the time_point and truncate it to seconds.
time_t
system_clock::to_time_t(const time_point& t)
{
    return time_t(duration_cast<seconds>(t.time_since_epoch()).count());
}

// Just turn the time_t into a count of seconds and construct a time_point with it.
system_clock::time_point
system_clock::from_time_t(time_t t)
{
    return system_clock::time_point(seconds(t));
}

// monotonic_clock

// Note, in this implementation monotonic_clock and high_resolution_clock
//   are the same clock.  They are both based on mach_absolute_time().
//   mach_absolute_time() * MachInfo.numer / MachInfo.denom is the number of
//   nanoseconds since the computer booted up.  MachInfo.numer and MachInfo.denom
//   are run time constants supplied by the OS.  This clock has no relationship
//   to the Gregorian calendar.  It's main use is as a high resolution timer.

// MachInfo.numer / MachInfo.denom is often 1 on the latest equipment.  Specialize
//   for that case as an optimization.
static
monotonic_clock::rep
monotonic_simplified()
{
    return mach_absolute_time();
}

static
double
compute_monotonic_factor()
{
    mach_timebase_info_data_t MachInfo;
    mach_timebase_info(&MachInfo);
    return static_cast<double>(MachInfo.numer) / MachInfo.denom;
}

static
monotonic_clock::rep
monotonic_full()
{
    static const double factor = compute_monotonic_factor();
    return static_cast<monotonic_clock::rep>(mach_absolute_time() * factor);
}

typedef monotonic_clock::rep (*FP)();

static
FP
init_monotonic_clock()
{
    mach_timebase_info_data_t MachInfo;
    mach_timebase_info(&MachInfo);
    if (MachInfo.numer == MachInfo.denom)
        return &monotonic_simplified;
    return &monotonic_full;
}

monotonic_clock::time_point
monotonic_clock::now()
{
    static FP fp = init_monotonic_clock();
    return time_point(duration(fp()));
}

}  // namespace chrono
}  // namespace boost

//----------------------------------------------------------------------------//
//                                POSIX                                     //
//----------------------------------------------------------------------------//
#elif defined(BOOST_CHRONO_POSIX_API)

#include <time.h>  // for clock_gettime

namespace boost
{
namespace chrono
{

  //system_clock::time_point system_clock::now()
  //{
  //  timeval tod;
  //  ::gettimeofday( &tod, 0 );
  //
  //  return time_point(duration(
  //    (static_cast<system_clock::rep>( tod.tv_sec ) * 1000000) + tod.tv_usec));
  //}

  #ifndef CLOCK_REALTIME
  # error <time.h> does not supply CLOCK_REALTIME
  #endif

  system_clock::time_point system_clock::now()
  {
    timespec ts;
    if ( ::clock_gettime( CLOCK_REALTIME, &ts ) )
    {
      boost::throw_exception(
        system::system_error( errno, system::system_category, "chrono::system_clock" ));
    }

    return time_point(duration(
      static_cast<system_clock::rep>( ts.tv_sec ) * 1000000000 + ts.tv_nsec));
  }

  system_clock::time_point system_clock::now(system::error_code & ec)
  {
    timespec ts;
    if ( ::clock_gettime( CLOCK_REALTIME, &ts ) )
    {
      ec.assign( errno, system::system_category );
      return time_point();
    }

    ec.clear();
    return time_point(duration(
      static_cast<system_clock::rep>( ts.tv_sec ) * 1000000000 + ts.tv_nsec));
  }

  std::time_t system_clock::to_time_t(const system_clock::time_point& t)
  {
      return static_cast<std::time_t>( t.time_since_epoch().count() / 1000000000 );
  }

  system_clock::time_point system_clock::from_time_t(std::time_t t)
  {
      return time_point(duration(static_cast<system_clock::rep>(t) * 1000000000));
  }

  #ifndef CLOCK_MONOTONIC
  # error <time.h> does not supply CLOCK_MONOTONIC
  #endif

  monotonic_clock::time_point monotonic_clock::now()
  {
    timespec ts;
    if ( ::clock_gettime( CLOCK_MONOTONIC, &ts ) )
    {
      boost::throw_exception(
        system::system_error( errno, system::system_category, "chrono::monotonic_clock" ));
    }

    return time_point(duration(
      static_cast<monotonic_clock::rep>( ts.tv_sec ) * 1000000000 + ts.tv_nsec));
  }

  monotonic_clock::time_point monotonic_clock::now(system::error_code & ec)
  {
    timespec ts;
    if ( ::clock_gettime( CLOCK_MONOTONIC, &ts ) )
    {
      ec.assign( errno, system::system_category );
      return time_point();
    }

    ec.clear();
    return time_point(duration(
      static_cast<monotonic_clock::rep>( ts.tv_sec ) * 1000000000 + ts.tv_nsec));
  }

}  // namespace chrono
}  // namespace boost

#endif  // POSIX
