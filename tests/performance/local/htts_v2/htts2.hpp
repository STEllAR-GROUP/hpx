//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013-2014 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <iostream>
#include <boost/format.hpp>

#include <boost/assert.hpp>

#include <cmath>
#include <boost/cstdint.hpp>

#include <chrono>
#include <boost/chrono.hpp>

namespace htts2
{

enum io_type
{
    csv_with_headers,
    csv_without_headers
};

template <typename BaseClock>
struct clocksource
{
    typedef BaseClock base_clock;
    typedef typename base_clock::duration duration;
    typedef typename base_clock::period period;
    typedef typename base_clock::rep rep;

    static_assert(base_clock::is_steady == true,
        "base_clock is not steady");
#if !defined(HPX_MSVC)
    static_assert(
            std::ratio_equal<period, std::nano>::value ||
            boost::ratio_equal<period, boost::nano>::value,
        "base_clock does not use a nanosecond period");
#endif

    // Returns: current time in nanoseconds.
    static rep now()
    {
        duration d = base_clock::now().time_since_epoch();
        rep t = d.count();
        BOOST_ASSERT(t >= 0);
        return t;
    }

    // Returns: uncertainty of the base_clock in nanoseconds.
    static double clock_uncertainty()
    {
        // For steady clocks, we use instrument uncertainty, ie:
        //   instrument_uncertainty = instrument_least_count/2
        return 1.0/2.0;
    }
};

// Performs approximately 'expected_' nanoseconds of artificial work.
// Returns: nanoseconds of work performed.
template <typename BaseClock>
typename clocksource<BaseClock>::rep
payload(typename clocksource<BaseClock>::rep expected)
{
    typedef typename clocksource<BaseClock>::rep rep;

    rep const start = clocksource<BaseClock>::now();

    while (true)
    {
        rep const measured = clocksource<BaseClock>::now() - start;

        if (measured >= expected)
            return measured;
    }
}

template <typename BaseClock = boost::chrono::steady_clock>
struct timer : clocksource<BaseClock>
{
    typedef typename clocksource<BaseClock>::rep rep;

    timer() : start_(clocksource<BaseClock>::now()) {}

    void restart()
    {
        start_ = this->now();
    }

    // Returns: elapsed time in nanoseconds.
    rep elapsed() const
    {
        return this->now() - start_;
    }

    // Returns: uncertainty of elapsed time.
    double elapsed_uncertainty() const
    {
        return this->clock_uncertainty();
    }

  private:
    rep start_;
};

struct driver
{
    // Parses the command line.
    driver(int argc, char** argv, bool allow_unregistered = false);

    virtual ~driver() {}

  protected:
    // Reads from the command line.
    boost::uint64_t osthreads_;
    boost::uint64_t tasks_;
    boost::uint64_t payload_duration_;
    io_type         io_;

    // hold on to command line
    int argc_;
    char** argv_;
    bool allow_unregistered_;
};

}

