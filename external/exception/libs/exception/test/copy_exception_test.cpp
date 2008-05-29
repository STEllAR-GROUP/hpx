//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/exception_ptr.hpp>
#include <boost/detail/lightweight_test.hpp>

struct
test_exception:
    std::exception
    {
    };

int
main()
    {
    boost::exception_ptr p = boost::copy_exception(test_exception());
    try
        {
        rethrow_exception(p);
        BOOST_TEST(false);
        }
    catch(
    test_exception & )
        {
        }
    catch(
    ... )
        {
        BOOST_TEST(false);
        }
    return boost::report_errors();
    }
