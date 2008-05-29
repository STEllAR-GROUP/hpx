//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//This program simulates errors on copying simple data files.
//It demonstrates how the proposed Boost exception library can be used.
//
//The documentation for boost::exception can be found at:
//
//  http://www.revergestudios.com/boost-exception/boost-exception.htm.
//
//The output from this program can vary depending on the compiler/OS combination.

#include <boost/exception.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <stdio.h>
#include <errno.h>
#include <string>
#include <iostream>

typedef boost::error_info<struct tag_errno,int> errno_info;
typedef boost::error_info<struct tag_file_stream,boost::weak_ptr<FILE> > file_stream_info;
typedef boost::error_info<struct tag_open_mode,std::string> open_mode_info; //The open mode of a failed fopen request.
typedef boost::error_info<struct tag_file_name,std::string> file_name_info; //The file name of a failed file operation.
typedef boost::error_info<struct tag_file_name_src,std::string> file_name_src_info; //The source file name of a failed copy operation.
typedef boost::error_info<struct tag_file_name_dst,std::string> file_name_dst_info; //The destination file name of a failed copy operation.
typedef boost::error_info<struct tag_function,std::string> function_info; //The name of the C function which reported the failure.

char const data[] = "example";
size_t const data_size = sizeof(data);

class
error: //Base for all exception objects we throw.
    public std::exception,
    public boost::exception
    {
    public:

    char const *
    what() const throw()
        {
        return boost::exception::what();
        }

    protected:

    ~error() throw()
        {
        }
    };

class open_error: public error { };
class read_error: public error { };
class write_error: public error { };

class fopen_error: public open_error { };
class fread_error: public read_error { };
class fwrite_error: public write_error { };


boost::shared_ptr<FILE>
my_fopen( char const * name, char const * mode )
    {
    if( FILE * f = ::fopen(name,mode) )
        return boost::shared_ptr<FILE>(f,fclose);
    else
        throw fopen_error() << BOOST_ERROR_INFO <<
            errno_info(errno) <<
            file_name_info(name) <<
            open_mode_info(mode) <<
            function_info("fopen");
    }

void
my_fread( void * buffer, size_t size, size_t count, boost::shared_ptr<FILE> const & stream )
    {
    assert(stream);
    if( count!=fread(buffer,size,count,stream.get()) || ferror(stream.get()) )
        throw fread_error() << BOOST_ERROR_INFO <<
            function_info("fread") <<
            errno_info(errno) <<
            file_stream_info(boost::weak_ptr<FILE>(stream));
    }

void
my_fwrite( void const * buffer, size_t size, size_t count, boost::shared_ptr<FILE> const & stream )
    {
    assert(stream);
    if( count!=fwrite(buffer,size,count,stream.get()) || ferror(stream.get()) )
        throw fwrite_error() << BOOST_ERROR_INFO <<
            function_info("fwrite") <<
            errno_info(errno) <<
            file_stream_info(boost::weak_ptr<FILE>(stream));
    }

void
reset_file( char const * file_name )
    {
    (void) my_fopen(file_name,"wb");
    }

void
create_data( char const * file_name )
    {
    boost::shared_ptr<FILE> f = my_fopen(file_name,"wb");
    my_fwrite( data, 1, data_size, f );
    }

void
copy_data( char const * src_file_name, char const * dst_file_name )
    {
    boost::shared_ptr<FILE> src = my_fopen(src_file_name,"rb");
    boost::shared_ptr<FILE> dst = my_fopen(dst_file_name,"wb");
    try
        {
        char buffer[data_size];
        my_fread( buffer, 1, data_size, src );
        my_fwrite( buffer, 1, data_size, dst );
        }
    catch(
    boost::exception & x )
        {
        if( boost::shared_ptr<boost::weak_ptr<FILE> const> f=boost::get_error_info<file_stream_info>(x) )
            if( boost::shared_ptr<FILE> fs = f->lock() )
                {
                if( fs==src )
                    x << file_name_info(src_file_name);
                else if( fs==dst )
                    x << file_name_info(dst_file_name);
                }
        x <<
            file_name_src_info(src_file_name) <<
            file_name_dst_info(dst_file_name);
        throw;
        }
    }

void
dump_copy_info( boost::exception const & x )
    {
    if( boost::shared_ptr<std::string const> src = boost::get_error_info<file_name_src_info>(x) )
        std::cout << "Source file name: " << *src << "\n";
    if( boost::shared_ptr<std::string const> dst = boost::get_error_info<file_name_dst_info>(x) )
        std::cout << "Destination file name: " << *dst << "\n";
    }

void
dump_file_info( boost::exception const & x )
    {
    if( boost::shared_ptr<std::string const> fn = boost::get_error_info<file_name_info>(x) )
        std::cout << "Source file name: " << *fn << "\n";
    }

void
dump_clib_info( boost::exception const & x )
    {
    if( boost::shared_ptr<int const> err=boost::get_error_info<errno_info>(x) )
        std::cout << "OS error: " << *err << "\n";
    if( boost::shared_ptr<std::string const> fn=boost::get_error_info<function_info>(x) )
        std::cout << "Failed function: " << *fn << "\n";
    }

void
dump_all_info( boost::exception const & x )
    {
    std::cout << "-------------------------------------------------\n";
    dump_copy_info(x);
    dump_file_info(x);
    dump_clib_info(x);
    std::cout << "\nOutput from what():\n";
    std::cout << x.what();
    }

int
main()
    {
    try
        {
        create_data( "tmp1.txt" );
        copy_data( "tmp1.txt", "tmp2.txt" ); //This should succeed.

        reset_file( "tmp1.txt" ); //Creates empty file.
        try
            {
            copy_data( "tmp1.txt", "tmp2.txt" ); //This should fail, tmp1.txt is empty.
            }
        catch(
        read_error & x )
            {
            std::cout << "\nCaught 'read_error' exception.\n";
            dump_all_info(x);
            }

        remove( "tmp1.txt" );
        remove( "tmp2.txt" );
        try
            {
            copy_data( "tmp1.txt", "tmp2.txt" ); //This should fail, tmp1.txt does not exist.
            }
        catch(
        open_error & x )
            {
            std::cout << "\nCaught 'open_error' exception.\n";
            dump_all_info(x);
            }
        }
    catch(
    boost::exception & x )
        {
        std::cout << "\nCaught unexpected boost::exception!\n";
        dump_all_info(x);
        }
    }
