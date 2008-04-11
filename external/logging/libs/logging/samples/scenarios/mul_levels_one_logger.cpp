/**
 Boost Logging library

 Author: John Torjo, www.torjo.com

 Copyright (C) 2007 John Torjo (see www.torjo.com for email)

 Distributed under the Boost Software License, Version 1.0.
    (See accompanying file LICENSE_1_0.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)

 See http://www.boost.org for updates, documentation, and revision history.
 See http://www.torjo.com/log2/ for more details
*/

/**
@example mul_levels_one_logger.cpp

@copydoc mul_levels_one_logger

@page mul_levels_one_logger mul_levels_one_logger.cpp Example

This usage:
- You have multiple levels (in this example: debug < info < error)
- You want to format the message before it's written 
  (in this example: prefix it by time, by index, and append newline to it)
- You have <b>one log</b>, which writes to several log destinations
  (in this example: the console, the debug output window, and a file)

In this example, all output will be written to the console, debug output window, and "out.txt" file.
It will look similar to this one:

@code
21:03.17.243 [1] this is so cool 1
21:03.17.243 [2] first error 2
21:03.17.243 [3] hello, world
21:03.17.243 [4] second error 3
21:03.17.243 [5] good to be back ;) 4
21:03.17.243 [6] third error 5
@endcode

*/


#include <boost/logging/format/named_write.hpp>
typedef boost::logging::named_logger<>::type logger_type;

#define L_(lvl) BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_level(), lvl )

BOOST_DEFINE_LOG_FILTER(g_log_level, boost::logging::level::holder ) // holds the application log level
BOOST_DEFINE_LOG(g_l, logger_type)

void test_mul_levels_one_logger() {
    // formatting    : time [idx] message \n
    // destinations  : console, file "out.txt" and debug window
    g_l()->writer().write("%time%($hh:$mm.$ss.$mili) [%idx%] |\n", "cout file(out.txt) debug");
    g_l()->mark_as_initialized();

    int i = 1;
    L_(debug) << "this is so cool " << i++;
    L_(error) << "first error " << i++;

    std::string hello = "hello", world = "world";
    L_(debug) << hello << ", " << world;

    using namespace boost::logging;
    g_log_level()->set_enabled(level::error);
    L_(debug) << "this will not be written anywhere";
    L_(info) << "this won't be written anywhere either";
    L_(error) << "second error " << i++;

    g_log_level()->set_enabled(level::info);
    L_(info) << "good to be back ;) " << i++;
    L_(error) << "third error " << i++;
}



int main() {
    test_mul_levels_one_logger();
}


// End of file

