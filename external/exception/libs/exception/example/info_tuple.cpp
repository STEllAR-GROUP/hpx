//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//This example shows how boost::error_info_group can be used to bundle
//the name of the function that fails together with the reported errno.

#include <boost/exception/info_tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <stdio.h>
#include <string>
#include <errno.h>

typedef boost::error_info<struct tag_file_name,std::string> file_name_info;
typedef boost::error_info<struct tag_function,char const *> function_info;
typedef boost::error_info<struct tag_errno,int> errno_info;
typedef boost::tuple<function_info,errno_info> clib_failure;

class file_open_error: public boost::exception { };

boost::shared_ptr<FILE>
file_open( char const * name, char const * mode )
    {
    if( FILE * f=fopen(name,mode) )
        return boost::shared_ptr<FILE>(f,fclose);
    else
        throw file_open_error() <<
            file_name_info(name) <<
            clib_failure("fopen",errno);
    }
