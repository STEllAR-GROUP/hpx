//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/exception/info.hpp>
#include <boost/detail/lightweight_test.hpp>

typedef boost::error_info<struct tag_test,int> test;

class
my_exception:
    public boost::exception
    {
    };

int
main()
    {
    my_exception x;
    x << test(1);
    std::string w1 = x.what();
    x << test(2);
    std::string w2 = x.what();
    BOOST_TEST( w1!=w2 );
    return boost::report_errors();
    }
