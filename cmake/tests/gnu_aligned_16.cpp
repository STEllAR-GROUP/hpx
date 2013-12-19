////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/detail/lightweight_test.hpp>

namespace { struct aligned_16 { char a; } __attribute__ ((aligned(16))); }

int main()
{
    BOOST_TEST_EQ(sizeof(aligned_16), 16U);
    return boost::report_errors();
}

