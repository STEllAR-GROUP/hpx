//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/exception_ptr.hpp>
#include <boost/exception/info.hpp>
#include <boost/detail/lightweight_test.hpp>

typedef boost::error_info<struct tag_test,int> test;

struct
test_boost_exception:
    boost::exception
    {
    };

void
throw_boost_exception()
    {
    throw test_boost_exception() << test(42);
    }

void
throw_unknown_exception()
    {
    struct
    test_exception:
        std::exception
        {
        };
    throw test_exception();
    }

int
main()
    {
    try
        {
        throw_boost_exception();
        }
    catch(
    ... )
        {
        boost::exception_ptr ep=boost::current_exception();
        try
            {
            rethrow_exception(ep);
            }
        catch(
        boost::unknown_exception & x )
            {
            BOOST_TEST( 42==*boost::get_error_info<test>(x) );
            }
        catch(
        ... )
            {
            BOOST_TEST(false);
            }
        try
            {
            rethrow_exception(ep);
            }
        catch(
        boost::exception & x )
            {
            BOOST_TEST( 42==*boost::get_error_info<test>(x) );
            }
        catch(
        ... )
            {
            BOOST_TEST(false);
            }
        }
    try
        {
        throw_unknown_exception();
        }
    catch(
    ... )
        {
        boost::exception_ptr ep=boost::current_exception();
        try
            {
            rethrow_exception(ep);
            }
        catch(
        boost::unknown_exception & )
            {
            }
        catch(
        ... )
            {
            BOOST_TEST(false);
            }
        try
            {
            rethrow_exception(ep);
            }
        catch(
        boost::exception & )
            {
            }
        catch(
        ... )
            {
            BOOST_TEST(false);
            }
        }
    return boost::report_errors();
    }
