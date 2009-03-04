//  timex: timed execution program  ------------------------------------------//

//  Copyright Beman Dawes 2007

//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org/libs/system for documentation.

#include <boost/system/timer.hpp>
#include <cstdlib>
#include <string>
#include <iostream>

int main( int argc, char * argv[] )
{
  if ( argc == 1 )
  {
    std::cout << "invoke: timex [-v] command [args...]\n"
      "  command will be executed and timings displayed\n"
      "  -v option causes command and args to be displayed\n";
    return 1;
  }

  std::string s;

  bool verbose = false;
  if ( argc > 1 && *argv[1] == '-' && *(argv[1]+1) == 'v' )
  {
    verbose = true;
    ++argv;
    --argc;
  }

  for ( int i = 1; i < argc; ++i )
  {
    if ( i > 1 ) s += ' ';
    s += argv[i];
  }

  if ( verbose )
    { std::cout << "command: \"" << s.c_str() << "\"\n"; }

  boost::system::run_timer t;

  return std::system( s.c_str() );
}
