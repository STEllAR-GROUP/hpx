////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <cmath>

#include <boost/function.hpp>
#include <boost/phoenix/core.hpp>
#include <boost/phoenix/function.hpp>
#include <boost/phoenix/operator.hpp>

using std::abs;
using std::cout;
using std::endl;
using std::setprecision;

using boost::function;
using boost::phoenix::placeholders::_1;

BOOST_PHOENIX_ADAPT_FUNCTION(double, sine, std::sin, 1)

double area(
    boost::function<double (double)> const& f
  , double x
  , double increment
) {
    return f(x)*increment;
}

double integrate(
    boost::function<double (double)> const& f
  , double lower_bound
  , double upper_bound
  , double tolerance
) {
    double total_area = 0;

    // Define default rectangle width to be 1% of the total range evaluated.
    const double default_increment = (0.01*(upper_bound-lower_bound));
    double temp_increment = default_increment;

    for (double i = lower_bound; i<upper_bound; i+=temp_increment) {
        // Reset increment at the start of each for-loop iteration to avoid
        // unneeded resolution.
        temp_increment = default_increment;
        // When the difference between the function value at the middle of the
        // increment and the start of the increment is too big, decrease the
        // increment by a factor of two and retry until increment reaches
        // minimum.
           while ((abs(f(i+(temp_increment/2))-f(i)))>tolerance) {
               temp_increment/=2;
               // Smallest allowed increment is 1/10000 of the range.
               if (temp_increment<(0.01*default_increment)) break;
           }
        total_area+=area(f, i, temp_increment);
    }

    return total_area;
}

int main (void)
{   
    double t = 0.0000000005;  // tolerance
    double a = 0;             // lower bound of integration
    double b = 2 * 3.1415926; // upper bound of integration

    // the integral of sin(x) * sin(x) from 0 to 2 pi is pi
    cout<<setprecision(12)<<integrate(sine(_1)*sine(_1),a,b,t)<<endl;
}

