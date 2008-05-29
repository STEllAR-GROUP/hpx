//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//This example shows how to add data to boost::exception objects at the
//point of the throw, and how to retrieve that data at the point of the catch.

#include <boost/exception.hpp>
#include <errno.h>
#include <iostream>

typedef boost::error_info<struct tag_errno,int> errno_info; //(1)

class my_error: public boost::exception, public std::exception { }; //(2)

void
f()
    {
    throw my_error() << errno_info(errno); //(3)
    }
                 
void
g()
    {
    try
        {
        f();
        }
    catch(
    my_error & x )
        {
        if( boost::shared_ptr<int const> err=boost::get_error_info<errno_info>(x) )
            std::cerr << "Error code: " << *err;
        }
    }
