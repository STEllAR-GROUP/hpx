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

// main.cpp : Where the main() action is

/**  
@page scenario_multiple_files A more complex example - a Line counter application

- @ref scenario_multiple_files_program 
- @ref scenario_multiple_files_log_h 
- @ref scenario_multiple_files_log_cpp 
- @ref scenario_multiple_files_main 



@section scenario_multiple_files_program What the program does

This represents a small program to count lines of code from a directory and its subdirs.
It's a simple application, just to show you how to use the Boost Logging Lib v2 in your own projects.

It counts code lines, empty lines, and comment lines.
Again, it's a very simple program, thus:
- an empty line : contains only spaces
- a comment line: 
  - C++ : starts with //
  - C: the start of a C comment is when the line @b starts with "/ *"; 
       the end of a C comment is when the line @b ends with "* /";
       anything else is not considered a comment


Command line arguments:
- 1st: the dir to search (default : ".")
- 2nd: the extensions to search (default: "cpp;c;h;hpp")
- 3rd: verbosity: "d" : debug, "i" : info, "e" : error (default : "i")


About logging:
- You have multiple levels (in this example: debug < info < error)
- Message is formatted like this: <tt>time [idx] message <enter> </tt>
- You have <b>one log</b>, which writes to several log destinations:
  the console, the debug output window, and the "out.txt" file
- all the logs are @b declared in "log.h" file
- all the logs are @b defined in "log.cpp" file

The highlights of setting up the logging are :
- @c log.h file - declaring the log/macros to use the log
- @c log.cpp file - defining the log, and the @c init_logs function (initializing the log)
- @c main.cpp file - reading the command line, and calls the @c init_logs function

You can check out the whole example: <tt>libs/logging/samples/basic_usage</tt>.

@section scenario_multiple_files_log_h The log.h file

@include basic_usage/log.h

@section scenario_multiple_files_log_cpp The log.cpp file

@include basic_usage/log.cpp

@section scenario_multiple_files_main The main.cpp file

@include basic_usage/main.cpp

*/

// Wherever you use logs, include this ;)
#include "log.h"

#include <string>
#include <sstream>
#include <iostream>

#include "dir_spec.h"
#include "extensions.h"
#include "util.h"
#include <boost/filesystem/path.hpp>

using namespace boost::logging;
namespace fs = boost::filesystem;

int main(int argc, char * argv[])  
{
    fs::path::default_name_check( fs::no_check);

    init_logs();

    std::string dir = ".";
    if ( argc > 1)
        dir = argv[1];

    extensions ext;
    {
    std::string ext_str = "cpp;c;h;hpp";
    if ( argc > 2)
        ext_str = argv[2];
    str_replace(ext_str, ";", " ");
    str_replace(ext_str, ",", " ");
    str_replace(ext_str, ".", "");
    std::istringstream in( ext_str);
    std::string word;
    while ( in >> word)
        ext.add(word);
    }
    
    level::type lev = level::info;
    std::string lev_str = "info";
    if ( argc > 3) {
        lev_str = argv[3];
        if ( lev_str == "d") {
            lev = level::debug;
            lev_str = "debug";
        }
        else if ( lev_str == "i") {
            lev = level::info;
            lev_str = "info";
        }
        else if ( lev_str == "e") {
            lev = level::error;
            lev_str = "error";
        }
        else {
            LERR_ << "invalid verbosity " << lev_str << "; available options: d, i, e";
            lev_str = "info";
        }
    }
    g_l_filter()->set_enabled(lev);

    LAPP_ << "Verbosity: " << lev_str;

    dir_spec spec(dir, ext);
    spec.iterate();
	return 0;
}

