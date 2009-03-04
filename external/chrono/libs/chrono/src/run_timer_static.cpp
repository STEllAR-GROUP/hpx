//  boost run_timer_static.cpp  ---------------------------------------------------//

//  Copyright Beman Dawes 2008

//  Distributed under the Boost Software License, Version 1.0.
//  See http://www.boost.org/LICENSE_1_0.txt

//  See http://www.boost.org/libs/chrono for documentation.

//---------------------------------------------------------------------------------//

//  This function is defined in a separate translation so that it will not be linked
//  in except if actually used. This is more efficient because header <iostream> is
//  required, and it incurs the cost of the standard stream objects even if they are
//  not actually used.

//---------------------------------------------------------------------------------//

// define BOOST_CHRONO_SOURCE so that <boost/chrono/config.hpp> knows
// the library is being built (possibly exporting rather than importing code)
#define BOOST_CHRONO_SOURCE 

#include <boost/chrono/process_times.hpp>
#include <iostream>

namespace boost
{
  namespace chrono
  {

    std::ostream &  run_timer::m_cout()  { return std::cout; }

  } // namespace chrono
} // namespace boost
