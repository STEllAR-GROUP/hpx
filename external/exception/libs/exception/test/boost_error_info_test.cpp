//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/exception/info.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <stdexcept>

namespace
test
    {
    class my_exception: public boost::exception { };

    typedef boost::error_info<struct tag_my_info,int> my_info;

    void
    test_boost_error_info()
        {
        try
            {
            throw my_exception() << BOOST_ERROR_INFO << my_info(1);
            }
        catch(
        my_exception & x )
            {
            BOOST_TEST(1==*boost::get_error_info<my_info>(x));
            BOOST_TEST(boost::get_error_info<boost::throw_function>(x));
            BOOST_TEST(boost::get_error_info<boost::throw_file>(x));
            BOOST_TEST(boost::get_error_info<boost::throw_line>(x));
            }
        }
    }

int
main()
    {
    test::test_boost_error_info();
    return boost::report_errors();
    }
