//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/exception/info_tuple.hpp>
#include <boost/detail/lightweight_test.hpp>

struct throws_on_copy;
struct non_printable { };

typedef boost::error_info<struct tag_test_1,int> test_1;
typedef boost::error_info<struct tag_test_2,unsigned int> test_2;
typedef boost::error_info<struct tag_test_3,float> test_3;
typedef boost::error_info<struct tag_test_4,throws_on_copy> test_4;
typedef boost::error_info<struct tag_test_5,std::string> test_5;
typedef boost::error_info<struct tag_test_6,non_printable> test_6;

struct
test_exception:
    public boost::exception
    {
    };

struct
throws_on_copy
    {
    throws_on_copy()
        {
        }

    throws_on_copy( throws_on_copy const & )
        {
        throw test_exception();
        }
    };

void
basic_test()
    {
    test_exception x;
    x << test_1(1) << test_2(2u) << test_3(3.14159f);
    BOOST_TEST(*boost::get_error_info<test_1>(x)==1);
    BOOST_TEST(*boost::get_error_info<test_2>(x)==2u);
    BOOST_TEST(*boost::get_error_info<test_3>(x)==3.14159f);
    BOOST_TEST(!boost::get_error_info<test_4>(x));
    }

void
exception_safety_test()
    {
    test_exception x;
    try
        {
        x << test_4(throws_on_copy());
        BOOST_TEST(false);
        }
    catch(
    test_exception & )
        {
        }
    BOOST_TEST(!boost::get_error_info<test_4>(x));
    }

void
throw_empty()
    {
    throw test_exception();
    }

void
throw_test_1( char const * value )
    {
    throw test_exception() << test_5(std::string(value));
    }

void
throw_test_2()
    {
    throw test_exception() << test_6(non_printable());
    }

void
throw_catch_add_file_name( char const * name )
    {
    try
        {
        throw_empty();
        BOOST_TEST(false);
        }
    catch(
    boost::exception & x )
        {
        x << test_5(std::string(name));
        throw;
        }     
    }

void
test_empty()
    {
    try
        {
        throw_empty();
        BOOST_TEST(false);
        }
    catch(
    boost::exception & x )
        {
        BOOST_TEST( dynamic_cast<test_exception *>(&x) );
        BOOST_TEST( !boost::get_error_info<test_1>(x) );
        }

    try
        {
        throw_empty();
        BOOST_TEST(false);
        }
    catch(
    test_exception & x )
        {
        BOOST_TEST( dynamic_cast<boost::exception *>(&x) );
        }
    }

void
test_basic_throw_catch()
    {
    try
        {
        throw_test_1("test");
        BOOST_ASSERT(false);
        }
    catch(
    boost::exception & x )
        {
        BOOST_TEST(*boost::get_error_info<test_5>(x)==std::string("test"));
        }
    try
        {
        throw_test_2();
        BOOST_ASSERT(false);
        }
    catch(
    boost::exception & x )
        {
        BOOST_TEST(boost::get_error_info<test_6>(x));
        }
    }

void
test_catch_add_info()
    {
    try
        {
        throw_catch_add_file_name("test");
        BOOST_TEST(false);
        }
    catch(
    boost::exception & x )
        {
        BOOST_TEST(*boost::get_error_info<test_5>(x)==std::string("test"));
        }
    }

void
test_add_tuple()
    {
    typedef boost::tuple<test_1,test_2> test_12;
    typedef boost::tuple<test_1,test_2,test_3> test_123;
    typedef boost::tuple<test_1,test_2,test_3,test_5> test_1235;
    try
        {
        throw test_exception() << test_12(42,42u);
        }
    catch(
    test_exception & x )
        {
        BOOST_TEST( *boost::get_error_info<test_1>(x)==42 );
        BOOST_TEST( *boost::get_error_info<test_2>(x)==42u );
        }
    try
        {
        throw test_exception() << test_123(42,42u,42.0f);
        }
    catch(
    test_exception & x )
        {
        BOOST_TEST( *boost::get_error_info<test_1>(x)==42 );
        BOOST_TEST( *boost::get_error_info<test_2>(x)==42u );
        BOOST_TEST( *boost::get_error_info<test_3>(x)==42.0f );
        }
    try
        {
        throw test_exception() << test_1235(42,42u,42.0f,std::string("42"));
        }
    catch(
    test_exception & x )
        {
        BOOST_TEST( *boost::get_error_info<test_1>(x)==42 );
        BOOST_TEST( *boost::get_error_info<test_2>(x)==42u );
        BOOST_TEST( *boost::get_error_info<test_3>(x)==42.0f );
        BOOST_TEST( *boost::get_error_info<test_5>(x)=="42" );
        }
    }

int
main()
    {
    basic_test();
    exception_safety_test();
    test_empty();
    test_basic_throw_catch();
    test_catch_add_info();
    test_add_tuple();
    return boost::report_errors();
    }
