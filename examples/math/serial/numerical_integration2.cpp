////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/function.hpp>
#include <boost/phoenix/core.hpp>
#include <boost/phoenix/function.hpp>
#include <boost/phoenix/operator.hpp>

#include <hpx/config.hpp>
#include <hpx/util/high_resolution_timer.hpp>

using std::abs;
using std::sin;
using std::cout;
using std::cerr;
using std::endl;
using std::setprecision;
using std::numeric_limits;
 
using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::store;
using boost::program_options::command_line_parser;
using boost::program_options::notify;

using boost::format;
using boost::io::group;

using boost::function;

using boost::math::constants::pi;

using boost::phoenix::placeholders::_1;

using hpx::util::high_resolution_timer;

enum
{
    d_precision = numeric_limits<double>::digits10 + 1,
    ld_precision = numeric_limits<long double>::digits10 + 1
};

BOOST_PHOENIX_ADAPT_FUNCTION(long double, sine, std::sin, 1)

long double integrate_fastpath(
    boost::function<long double(long double)> const& f
  , long double lower_bound
  , long double upper_bound
  , long double tolerance
  , long double increment
) {
    long double total_area = 0.0L, temp_increment = increment;

    boost::uint64_t last_count = 0;

    for (long double i = lower_bound; i < upper_bound;)
    {
        const long double fi = f(i);

        boost::uint64_t count = 0;

        // When the difference between the function value at the middle of the
        // increment and the start of the increment is too big, decrease the
        // increment by a factor of two and retry.
        while ((abs(f(i + (temp_increment / 2.0L)) - fi)) > tolerance)
        {
            ++count;
            temp_increment /= 2.0L; // TODO: ensure that I am optimized away,
                                    // as I am computed at the head of the while
                                    // loop.
        }

        total_area += fi * temp_increment;
        i += temp_increment;

        last_count = count;

        // Rollback one level of resolution at the end of each for-loop
        // iteration to avoid unneeded resolution, if we were not within the 
        // tolerance.
        if (count) 
            temp_increment *= 2.0L; 
    }

    return total_area;
}

long double integrate_slowpath(
    boost::function<long double(long double)> const& f
  , long double lower_bound
  , long double upper_bound
  , long double tolerance
  , long double increment
) {
    long double total_area = 0.0L, temp_increment = increment;

    boost::uint64_t count_depth = 0
                  , last_count = 0
                  , iterations = 0
                  , rollbacks = 0
                  , refinements = 0;

    for (long double i = lower_bound; i < upper_bound; ++iterations)
    {
        const long double fi = f(i);

        boost::uint64_t count = 0;

        // When the difference between the function value at the middle of the
        // increment and the start of the increment is too big, decrease the
        // increment by a factor of two and retry.
        while ((abs(f(i + (temp_increment / 2.0L)) - fi)) > tolerance)
        {
            ++count;
            temp_increment /= 2.0L; // TODO: ensure that I am optimized away,
                                    // as I am computed at the head of the while
                                    // loop.
        }

        if (count != last_count)
            cout << ( format("growth rate of increment changed from 1/2^%1% to "
                             "1/2^%2% at f(%3%)\n")
                    % (last_count + count_depth)
                    % (count + count_depth)
                    % group(setprecision(ld_precision), i));

        refinements += count;

        last_count = count;
        count_depth += count;

        total_area += fi * temp_increment;
        i += temp_increment;

        // Rollback one level of resolution at the end of each for-loop
        // iteration to avoid unneeded resolution, if we were not within the 
        // tolerance.
        if (count) 
        {
            ++rollbacks;
            --count_depth;
            temp_increment *= 2.0L;
        }
    }

    cout << ( format("computation completed in %1% iterations\n"
                     "%2% refinements occurred\n"
                     "%3% rollbacks were performed\n")
            % iterations
            % refinements
            % rollbacks);

    return total_area;
}

enum implementation
{
    fastpath = 0,
    slowpath = 1
};

int main(int argc, char** argv)
{
    variables_map vm;

    options_description
        desc_cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
   
    const long double pi_ = pi<long double>();

    long double tolerance(0.0L)
              , lower_bound(0.0L)
              , upper_bound(0.0L)
              , increment(0.0L); 

    desc_cmdline.add_options()
        ( "help,h"
        , "print out program usage (this message)")

        ( "tolerance"
        , value<long double>(&tolerance)->default_value(5.0e-08L, "5e-08") 
        , "resolution tolerance")

        ( "lower-bound"
        , value<long double>(&lower_bound)->default_value(0.0L, "0") 
        , "lower bound of integration")

        ( "upper-bound"
        , value<long double>(&upper_bound)->default_value(2.0L * pi_, "2*pi")
        , "upper bound of integration")

        ( "increment"
        , value<long double>(&increment)->default_value(pi_, "pi") 
        , "initial integration increment")

        ( "path"
        , value<std::string>()->default_value("fast")
        , "select implementation: options are `fast' (no statistics/debug "
          "info) or `slow' (statistics and debug info collected and displayed)")
    ;

    store(command_line_parser(argc, argv).options(desc_cmdline).run(), vm);

    notify(vm);

    // print help screen
    if (vm.count("help"))
    {
        cout << desc_cmdline;
        return 0;
    }

    implementation impl;

    if (vm["path"].as<std::string>() == "fast")
        impl = fastpath; 

    else if (vm["path"].as<std::string>() == "slow")
        impl = slowpath; 

    else
    {
        cerr << ( format("error: unknown implementation '%1%'\n")
                % vm["path"].as<std::string>());
        return 1;
    }

    long double r(0.0L);
    double elapsed(0.0);

    if (fastpath == impl)
    {
        high_resolution_timer t;

        r = integrate_fastpath(sine(_1) * sine(_1)
                             , lower_bound
                             , upper_bound
                             , tolerance
                             , increment);

        elapsed = t.elapsed();
    }

    else if (slowpath == impl)
    {
        high_resolution_timer t;

        r = integrate_slowpath(sine(_1) * sine(_1)
                             , lower_bound
                             , upper_bound
                             , tolerance
                             , increment);

        elapsed = t.elapsed();
    }

    // the integral of sin(x) * sin(x) from 0 to 2 pi is pi
    cout << ( format("%1% integral of sin(x) * sin(x) from %2% to %3% is %4%\n"
                     "computation took %5% seconds\n")
            % ((fastpath == impl) ? "fastpath" : "slowpath")
            % group(setprecision(ld_precision), lower_bound)
            % group(setprecision(ld_precision), upper_bound)
            % group(setprecision(ld_precision), r)
            % group(setprecision(d_precision), elapsed));
}

