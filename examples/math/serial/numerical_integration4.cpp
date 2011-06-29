////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cfloat>
#include <limits>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/math/constants/constants.hpp>
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
using std::floor;
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

BOOST_PHOENIX_ADAPT_FUNCTION(long double, sine, sin, 1)

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
  , bool toplevel = false
) {
    long double total_area = 0.0L;

    boost::uint64_t iterations = 0;

    for (long double i = lower_bound; i < upper_bound;)
    {
        if (toplevel)
        {
            if (10 == iterations)
            {
                cout << ( format("at %1%/%2%\n")
                        % group(setprecision(ld_precision), i)
                        % group(setprecision(ld_precision), upper_bound));
                iterations = 0;
            }

            else
                ++iterations;
        }

        const long double fi = f(i);
        const long double next_i = i + increment;

        // If the increment is smaller than or equal to epsilon, give up.
        if (LDBL_EPSILON >= increment)
            total_area += fi * increment;

        // When the difference between the function value at the middle of the
        // increment and the start of the increment is too larger than the
        // tolerance, regrid this interval. The increment of the regrid is
        // the current increment divided by the refinement rate.
        else if ((abs(f(i + (increment / 2.0L)) - fi)) > tolerance)
        {
            long double next_increment = increment / refinement_rate;

            // If the next increment would be below epsilon, use epsilon as the
            // next increment. 
            if (LDBL_EPSILON >= next_increment) 
                next_increment = LDBL_EPSILON;

            // Account for inprecise division.
            const long double end = next_increment * refinement_rate;
            const long double end_i = i + end;

            total_area += integrate_fastpath
                (f, i, end_i, tolerance, next_increment, refinement_rate);

            // If the difference between the end of the block handled by the
            // recursive call and the end of the current iteration is greater
            // than epsilon, make another recursive call to handle the leftover
            // region. Don't bother with this if the next increment is epsilon. 
            if (LDBL_EPSILON != next_increment)
                if (!equal(end_i, next_i, LDBL_EPSILON))
                    total_area += integrate_fastpath
                        ( f, end_i, next_i, tolerance, next_i - end_i
                        , refinement_rate); 
        }

        else
            total_area += fi * increment;

        i = next_i; 
    }

    return total_area;
}

BOOST_PHOENIX_ADAPT_FUNCTION
    (long double, integrate_fastpath_, integrate_fastpath, 6)

long double integrate_slowpath(
    boost::function<long double(long double)> const& f
  , long double lower_bound
  , long double upper_bound
  , long double tolerance
  , long double increment
  , long double refinement_rate
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
        const long double next_i = i + increment;

        cout << (format("f(%1%) is %2%\n")
                % group(setprecision(ld_precision), i)
                % group(setprecision(ld_precision), fi));

        // If the increment is smaller than or equal to epsilon, give up.
        if (LDBL_EPSILON >= increment)
            total_area += fi * increment;

        // When the difference between the function value at the middle of the
        // increment and the start of the increment is larger than the
        // tolerance, regrid this interval. The increment of the regrid is
        // the current increment divided by the refinement rate.
        else if ((abs(f(i + (increment / 2.0L)) - fi)) > tolerance)
        {
            long double next_increment = increment / refinement_rate;

            // If the next increment would be below epsilon, use epsilon as the
            // next increment. 
            if (LDBL_EPSILON >= next_increment) 
                next_increment = LDBL_EPSILON;

            // Account for inprecise division.
            const long double end = next_increment * refinement_rate;
            const long double end_i = i + end;

            cout << (format("computing block, end_i is %1%, next_i is %2%\n")
                    % group(setprecision(ld_precision), end_i)
                    % group(setprecision(ld_precision), next_i));

            total_area += integrate_slowpath
                (f, i, end_i, tolerance, next_increment, refinement_rate, name);

            // If the difference between the end of the block handled by the
            // recursive call and the end of the current iteration is greater
            // than epsilon, make another recursive call to handle the leftover
            // region. Don't bother with this if the next increment is epsilon. 
            if (LDBL_EPSILON != next_increment)
                if (!equal(end_i, next_i, LDBL_EPSILON))
                {
                    cout << "computing end\n";
                    total_area += integrate_slowpath
                        ( f, end_i, next_i, tolerance, next_i - end_i
                        , refinement_rate, name); 
                }
        }

        else
            total_area += fi * increment;

        i = next_i; 
    }

    return total_area;
}

BOOST_PHOENIX_ADAPT_FUNCTION
    (long double, integrate_slowpath_, integrate_slowpath, 7)

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
              , x_lower_bound(0.0L)
              , x_upper_bound(0.0L)
              , y_lower_bound(0.0L)
              , y_upper_bound(0.0L)
              , increment(0.0L)
              , refinement_rate(0.0L);

    desc_cmdline.add_options()
        ( "help,h"
        , "print out program usage (this message)")

        ( "tolerance"
        , value<long double>(&tolerance)->default_value(0.1, "0.1") 
        , "resolution tolerance")

        ( "x-lower-bound"
        , value<long double>(&x_lower_bound)->default_value(0, "0") 
        , "lower bound of integration with respect to x")

        ( "x-upper-bound"
        , value<long double>(&x_upper_bound)->default_value(128 * pi_, "128*pi")
        , "upper bound of integration with respect to x")

        ( "y-lower-bound"
        , value<long double>(&y_lower_bound)->default_value(0, "0") 
        , "lower bound of integration with respect to y")

        ( "y-upper-bound"
        , value<long double>(&y_upper_bound)->default_value(128 * pi_, "128*pi")
        , "upper bound of integration with respect to y")

        ( "increment"
        , value<long double>(&increment)->default_value(0.1, "0.1") 
        , "initial integration increment")

        ( "refinement-rate"
        , value<long double>(&refinement_rate)->default_value(128, "128") 
        , "factor by which the increment is decreased when refinement occurs")

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
                       sine(_a * _a * _1 * _1 * val(1.0 / (1 << 30))) + 1
                   ]
                 , val(y_lower_bound)
                 , val(y_upper_bound)
                 , val(tolerance)
                 , val(increment)
                 , val(refinement_rate))
            , x_lower_bound
            , x_upper_bound
            , tolerance
            , increment
            , refinement_rate
            , true);

        elapsed = t.elapsed();
    }

    else if (slowpath == impl)
    {
        high_resolution_timer t;

        r = integrate_slowpath(
              integrate_slowpath_(
                   lambda(_a = val(_1))
                   [
                       sine(_a * _a * _1 * _1 * val(1.0 / (1 << 30))) + 1
                   ]
                 , val(y_lower_bound)
                 , val(y_upper_bound)
                 , val(tolerance)
                 , val(increment)
                 , val(refinement_rate)
                 , val("dy"))
            , x_lower_bound
            , x_upper_bound
            , tolerance
            , increment
            , refinement_rate
            , "dx");

        elapsed = t.elapsed();
    }

    // With the default values, should be around 198566.0
    cout << ( format("%1% integral sin(1/2^30 * x^2 * y^2) + 1 with\n"
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

