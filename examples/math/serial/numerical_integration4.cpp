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
#include <boost/function.hpp>
#include <boost/phoenix/core.hpp>
#include <boost/phoenix/bind.hpp>
#include <boost/phoenix/scope.hpp>
#include <boost/phoenix/function.hpp>
#include <boost/phoenix/operator.hpp>

#include <hpx/config.hpp>
#include <hpx/util/high_resolution_timer.hpp>

using std::abs;
using std::sin;
using std::log;
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

using boost::phoenix::bind;
using boost::phoenix::val;
using boost::phoenix::lambda;
using boost::phoenix::local_names::_a;
using boost::phoenix::placeholders::_1;

using hpx::util::high_resolution_timer;

enum
{
    d_precision = numeric_limits<double>::digits10 + 1,
    ld_precision = numeric_limits<long double>::digits10 + 1
};

inline bool equal(long double x1, long double x2, long double epsilon) 
{
    // the numbers are close enough
    if (x1 + epsilon >= x2 && x1 - epsilon <= x2)
        return true;
    else
        return false;
}

long double integrate_fastpath(
    boost::function<long double(long double)> const& f
  , long double lower_bound
  , long double upper_bound
  , long double tolerance
  , long double increment
  , long double refinement_rate
  , long double minimum_increment
) {
    long double total_area = 0.0L;

    for (long double i = lower_bound; i < upper_bound;)
    {
        const long double fi = f(i);
        const long double next_i = i + increment;

        // If we're smaller than or equal to the minimum_increment, don't refine
        // any further.
        //if (minimum_increment >= increment)
        //    total_area += fi * increment; 

        // When the difference between the function value at the middle of the
        // increment and the start of the increment is too larger than the
        // tolerance, regrid this interval. The increment of the regrid is
        // the current increment divided by the refinement rate.
        /*else*/ if ((abs(f(i + (increment / 2.0L)) - fi)) > tolerance)
        {
            long double next_increment = increment / refinement_rate;

            // Figure out how many iterations can be done with the next
            // increment. Round it down, and call integrate recursively to
            // integrate that block. 
            const long double iterations = (next_i - i) / next_increment;
            const long double end = increment * std::floor(iterations);

            total_area += integrate_fastpath
                ( f, i, end, tolerance, next_increment
                , refinement_rate, minimum_increment);

            // If the difference between the end of the block handled by the
            // recursive call and the end of the current iteration is greater
            // than 1.0e-8L, make another recursive call to handle the leftover
            // region. 
            if (!equal(end, next_i, 1.0e-8L))
                total_area += integrate_fastpath
                    ( f, end, next_i, tolerance, next_i - end
                    , refinement_rate, minimum_increment); 
        }

        else
            total_area += fi * increment;

        i = next_i; 
    }

    return total_area;
}

BOOST_PHOENIX_ADAPT_FUNCTION
    (long double, integrate_fastpath_, integrate_fastpath, 7)

long double integrate_slowpath(
    boost::function<long double(long double)> const& f
  , long double lower_bound
  , long double upper_bound
  , long double tolerance
  , long double increment
  , long double refinement_rate
  , long double minimum_increment
  , char const* name
) {
    long double total_area = 0.0L;

    cout << ( format("%1% from %2% to %3% with increment %4%\n")
            % name
            % group(setprecision(ld_precision), lower_bound)
            % group(setprecision(ld_precision), upper_bound)
            % group(setprecision(ld_precision), increment));

    for (long double i = lower_bound; i < upper_bound;)
    {
        const long double fi = f(i);
        long double next_i = i + increment;

        // If we're smaller than or equal to the minimum_increment, don't refine
        // any further.
        //if (minimum_increment >= increment)
        //    total_area += fi * increment; 

        // When the difference between the function value at the middle of the
        // increment and the start of the increment is too larger than the
        // tolerance, regrid this interval. The increment of the regrid is
        // the current increment divided by the refinement rate.
        /*else*/ if ((abs(f(i + (increment / 2.0L)) - fi)) > tolerance)
        {
            long double next_increment = increment / refinement_rate;

            // Figure out how many iterations can be done with the next
            // increment. Round it down, and call integrate recursively to
            // integrate that block. 
            const long double iterations = (next_i - i) / next_increment;
            const long double end = increment * std::floor(iterations);

            total_area += integrate_slowpath
                ( f, i, end, tolerance, next_increment
                , refinement_rate, minimum_increment, name);

            // If the difference between the end of the block handled by the
            // recursive call and the end of the current iteration is greater
            // than 1.0e-8L, make another recursive call to handle the leftover
            // region. 
            if (!equal(end, next_i, 1.0e-8L))
                total_area += integrate_slowpath
                    ( f, end, next_i, tolerance, next_i - end
                    , refinement_rate, minimum_increment, name); 
        }

        else
            total_area += fi * increment;

        i = next_i; 
    }

    return total_area;
}

BOOST_PHOENIX_ADAPT_FUNCTION
    (long double, integrate_slowpath_, integrate_slowpath, 8)

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
   
    long double tolerance(0.0L)
              , x_lower_bound(0.0L)
              , x_upper_bound(0.0L)
              , y_lower_bound(0.0L)
              , y_upper_bound(0.0L)
              , increment(0.0L)
              , refinement_rate(0.0L)
              , minimum_increment(0.0L);

    desc_cmdline.add_options()
        ( "help,h"
        , "print out program usage (this message)")

        ( "tolerance"
        , value<long double>(&tolerance)->default_value(1e-6, "1e-6") 
        , "resolution tolerance")

        ( "x-lower-bound"
        , value<long double>(&x_lower_bound)->default_value(11, "11") 
        , "lower bound of integration with respect to x")

        ( "x-upper-bound"
        , value<long double>(&x_upper_bound)->default_value(14, "14")
        , "upper bound of integration with respect to x")

        ( "y-lower-bound"
        , value<long double>(&y_lower_bound)->default_value(7, "7") 
        , "lower bound of integration with respect to y")

        ( "y-upper-bound"
        , value<long double>(&y_upper_bound)->default_value(10, "10")
        , "upper bound of integration with respect to y")

        ( "increment"
        , value<long double>(&increment)->default_value(0.1, "0.1") 
        , "initial integration increment")

        ( "refinement-rate"
        , value<long double>(&refinement_rate)->default_value(256, "256") 
        , "factor by which the increment is decreased when refinement occurs")

        ( "minimum-increment"
        , value<long double>(&minimum_increment)->default_value(1e-10, "1e-10") 
        , "smallest allowed increment")

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

        r = integrate_fastpath(
              integrate_fastpath_(
                   lambda(_a = val(_1))
                   [
                       _a * _a + val(4) * _1
                   ]
                 , val(y_lower_bound)
                 , val(y_upper_bound)
                 , val(tolerance)
                 , val(increment)
                 , val(refinement_rate)
                 , val(minimum_increment))
            , x_lower_bound
            , x_upper_bound
            , tolerance
            , increment
            , refinement_rate
            , minimum_increment);

        elapsed = t.elapsed();
    }

    else if (slowpath == impl)
    {
        high_resolution_timer t;

        r = integrate_slowpath(
              integrate_slowpath_(
                   lambda(_a = val(_1))
                   [
                       _a * _a + val(4) * _1
                   ]
                 , val(y_lower_bound)
                 , val(y_upper_bound)
                 , val(tolerance)
                 , val(increment)
                 , val(refinement_rate)
                 , val(minimum_increment)
                 , val("dy"))
            , x_lower_bound
            , x_upper_bound
            , tolerance
            , increment
            , refinement_rate
            , minimum_increment
            , "dx");

        elapsed = t.elapsed();
    }

    cout << ( format("%1% integral x^2 + 4y with\n"
                     "  x from %2% to %3%\n"
                     "  y from %4% to %5%\n"
                     "is %6%\n"
                     "computation took %7% seconds\n")
            % ((fastpath == impl) ? "fastpath" : "slowpath")
            % group(setprecision(ld_precision), x_lower_bound)
            % group(setprecision(ld_precision), x_upper_bound)
            % group(setprecision(ld_precision), y_lower_bound)
            % group(setprecision(ld_precision), y_upper_bound)
            % group(setprecision(ld_precision), r)
            % group(setprecision(d_precision), elapsed));
}

