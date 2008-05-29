//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "helper2.hpp"
#include <boost/exception/info.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/detail/lightweight_test.hpp>

typedef boost::error_info<struct tag_test_int,int> test_int;

void
throw_fwd( void (*thrower)(int) )
    {
    try
        {
        thrower(42);
        }
    catch(
    boost::exception & x )
        {
        x << test_int(42);
        throw;
        }
    }

template <class T>
void
tester()
    {
    try
        {
        throw_fwd( &boost::exception_test::throw_test_exception<T> );
        BOOST_ASSERT(false);
        }
    catch(
    ... )
        {
        boost::exception_ptr p = boost::current_exception();
        try
            {
            rethrow_exception(p);
            BOOST_TEST(false);
            }
        catch(
        T & y )
            {
            BOOST_TEST(*boost::get_error_info<test_int>(y)==42);
            BOOST_TEST(y.x_==42);
            }
        catch(
        ... )
            {
            BOOST_TEST(false);
            }
        }
    }

int
main()
    {
    tester<boost::exception_test::some_boost_exception>();
    tester<boost::exception_test::some_std_exception>();
    return boost::report_errors();
    }
