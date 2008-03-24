//  Copyright (c) 2005-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_HIGH_RESOLUTION_TIMER_MAR_24_2008_1222PM)
#define HPX_UTIL_HIGH_RESOLUTION_TIMER_MAR_24_2008_1222PM

#include <boost/config.hpp>
#include <boost/throw_exception.hpp>

#if defined(BOOST_HAS_UNISTD_H)
#include <unistd.h>
#endif

#if defined(BOOST_WINDOWS)

#include <stdexcept>
#include <limits>
#include <windows.h>

namespace hpx { namespace util 
{
    ///////////////////////////////////////////////////////////////////////////////
    //
    //  high_resolution_timer 
    //      A timer object measures elapsed time.
    //      CAUTION: Windows only!
    //
    ///////////////////////////////////////////////////////////////////////////////
    class high_resolution_timer
    {
    public:
        high_resolution_timer() 
        {
            start_time.QuadPart = 0;
            frequency.QuadPart = 0;

            if (!QueryPerformanceFrequency(&frequency))
                boost::throw_exception(std::runtime_error("Couldn't acquire frequency"));

            restart(); 
        } 

        high_resolution_timer(double t) 
        {
            start_time.QuadPart = 0;
            frequency.QuadPart = 0;

            if (!QueryPerformanceFrequency(&frequency))
                boost::throw_exception(std::runtime_error("Couldn't acquire frequency"));

            start_time.QuadPart = (LONGLONG)(t * frequency.QuadPart); 
        } 

        high_resolution_timer(high_resolution_timer const& rhs) 
          : start_time(rhs.start_time), frequency(rhs.frequency)
        {
        } 

        void restart() 
        { 
            if (!QueryPerformanceCounter(&start_time))
                boost::throw_exception(std::runtime_error("Couldn't initialize start_time"));
        } 
        double elapsed() const                  // return elapsed time in seconds
        { 
            LARGE_INTEGER now;
            if (!QueryPerformanceCounter(&now))
                boost::throw_exception(std::runtime_error("Couldn't get current time"));

            return double(now.QuadPart - start_time.QuadPart) / frequency.QuadPart;
        }

        double elapsed_max() const   // return estimated maximum value for elapsed()
        {
            return double((std::numeric_limits<LONGLONG>::max)() - start_time.QuadPart) / 
                double(frequency.QuadPart); 
        }

        double elapsed_min() const            // return minimum value for elapsed()
        { 
            return 1.0 / frequency.QuadPart; 
        }

    private:
        LARGE_INTEGER start_time;
        LARGE_INTEGER frequency;
    }; 

}} // namespace hpx::util

#elif defined(_POSIX_TIMERS)

#include <time.h>

namespace hpx { namespace util
{

    ///////////////////////////////////////////////////////////////////////////////
    //
    //  high_resolution_timer 
    //      A timer object measures elapsed time.
    //      CAUTION: Linux only!
    //
    ///////////////////////////////////////////////////////////////////////////////
    class high_resolution_timer
    {
    public:
        high_resolution_timer() 
        {
            start_time.tv_sec = 0;
            start_time.tv_nsec = 0;

            restart(); 
        } 

        high_resolution_timer(double t) 
        {
            start_time.tv_sec = time_t(t);
            start_time.tv_nsec = (t - start_time.tv_sec) * 10e9;
        }
        
        high_resolution_timer(high_resolution_timer const& rhs) 
          : start_time(rhs.start_time)
        {
        } 

        void restart() 
        { 
            if (-1 == clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start_time))
                boost::throw_exception(std::runtime_error("Couldn't initialize start_time"));
        } 
        double elapsed() const                  // return elapsed time in seconds
        { 
            timespec now;
            if (-1 == clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now))
                boost::throw_exception(std::runtime_error("Couldn't get current time"));

            if (now.tv_sec == start_time.tv_sec)
                return double(now.tv_nsec - start_time.tv_nsec) * 10e-9;
                
            return double(now.tv_sec - start_time.tv_sec) + 
                (double(now.tv_nsec - start_time.tv_nsec) * 10e-9);
        }

        double elapsed_max() const   // return estimated maximum value for elapsed()
        {
            return double((std::numeric_limits<time_t>::max)() - start_time.tv_sec); 
        }

        double elapsed_min() const            // return minimum value for elapsed()
        { 
            timespec resolution;
            if (-1 == clock_getres(CLOCK_THREAD_CPUTIME_ID, &resolution))
                boost::throw_exception(std::runtime_error("Couldn't get resolution"));
            return double(resolution.tv_sec + resolution.tv_nsec * 10e-9); 
        }

    private:
        timespec start_time;
    }; 

}} // namespace hpx::util

#else // !defined(BOOST_WINDOWS) && !defined(_POSIX_TIMERS)

//  For platforms other than Windows or Linux, simply fall back to boost::timer
#include <boost/timer.hpp>

namespace hpx { namespace util
{
    typedef boost::timer high_resolution_timer;
}}

#endif

#endif  

