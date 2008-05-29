//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//This example shows how to add arbitrary data to active exception objects.

#include <boost/exception.hpp>
#include <boost/shared_ptr.hpp>
#include <stdio.h>
#include <errno.h>

//

typedef boost::error_info<struct tag_errno,int> errno_info;

class file_read_error: public boost::exception { };

void
file_read( FILE * f, void * buffer, size_t size )
    {
    if( size!=fread(buffer,1,size,f) )
        throw file_read_error() << errno_info(errno);
    }

//

typedef boost::error_info<struct tag_file_name,std::string> file_name_info;

boost::shared_ptr<FILE> file_open( char const * file_name, char const * mode );
void file_read( FILE * f, void * buffer, size_t size );

void
parse_file( char const * file_name )
    {
    boost::shared_ptr<FILE> f = file_open(file_name,"rb");
    assert(f);
    try
        {
        char buf[1024];
        file_read( f.get(), buf, sizeof(buf) );
        }
    catch(
    boost::exception & e )
        {
        e << file_name_info(file_name);
        throw;
        }
    }
