////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <cmath>

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
using std::endl;
using std::setprecision;
 
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

BOOST_PHOENIX_ADAPT_FUNCTION(double, sine, sin, 1)

double integrate(
    function<double(double)> const& f
  , double lower_bound
  , double upper_bound
  , double tolerance
  , double default_increment
) {
    double total_area = 0;

    double temp_increment = default_increment;

    for (double i = lower_bound; i < upper_bound; i += temp_increment) {
        // Reset increment at the start of each for-loop iteration to avoid
        // unneeded resolution.
        temp_increment = default_increment;

        const double fi = f(i);

        // When the difference between the function value at the middle of the
        // increment and the start of the increment is too big, decrease the
        // increment by a factor of two and retry until increment reaches
        // minimum.
        while ((abs(f(i + (temp_increment / 2)) - fi)) > tolerance)
            temp_increment /= 2;

        total_area += fi * temp_increment;
    }

    return total_area;
}

int main(int argc, char** argv)
{
    cout << setprecision(12);

    variables_map vm;

    options_description
        desc_cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
   
    const double pi_ = pi<double>();

    double tolerance(0), lower_bound(0), upper_bound(0), increment(0); 

    desc_cmdline.add_options()
        ("help,h", "print out program usage (this message)")
        ("tolerance", value<double>(&tolerance)->default_value(0.00000005), 
         "resolution tolerance")
        ("lower-bound", value<double>(&lower_bound)->default_value(0), 
         "lower bound of integration")
        ("upper-bound", value<double>(&upper_bound)->default_value(2 * pi_), 
         "upper bound of integration")
        ("increment", value<double>(&increment)->default_value(0.02 * pi_), 
         "initial increment")
    ;

    store(command_line_parser(argc, argv).options(desc_cmdline).run(), vm);

    notify(vm);

    // print help screen
    if (vm.count("help"))
    {
        cout << desc_cmdline;
        return 0;
    }

    high_resolution_timer t;

    // the integral of sin(x) * sin(x) from 0 to 2 pi is pi
    double r = integrate(sine(_1) * sine(_1)
                       , lower_bound
                       , upper_bound
                       , tolerance
                       , increment);

    double elapsed = t.elapsed();

    cout << ( format("integral of sin(x)*sin(x) from %1% to %2% is %3%\n"
                     "computation took %4% seconds\n")
            % lower_bound % upper_bound % r % elapsed);
}

