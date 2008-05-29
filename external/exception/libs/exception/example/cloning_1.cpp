//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//This example shows how to enable cloning when throwing a boost::exception.

#include <boost/exception/enable_current_exception.hpp>
#include <boost/exception/info.hpp>
#include <stdio.h>
#include <errno.h>

typedef boost::error_info<struct tag_errno,int> errno_info;

class file_read_error: public boost::exception { };

void
file_read( FILE * f, void * buffer, size_t size )
    {
    if( size!=fread(buffer,1,size,f) )
        throw boost::enable_current_exception(file_read_error()) <<
            errno_info(errno);
    }
