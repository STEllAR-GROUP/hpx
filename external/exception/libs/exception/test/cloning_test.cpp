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
    try
        {
        throw boost::enable_current_exception(test_exception());
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
        test_exception & )
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
