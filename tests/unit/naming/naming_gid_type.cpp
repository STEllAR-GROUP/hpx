////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/detail/lightweight_test.hpp>
#include <hpx/runtime/naming/name.hpp>

int main()
{
    using hpx::naming::gid_type;

    // constructor tests 
    gid_type gid0(0xdeadbeefULL); // lsb ctor
    gid_type gid1(0xdeadbeefULL, 0xcededeedULL); // msb + lsb ctor

    BOOST_TEST_EQ(gid0.get_lsb(), 0xdeadbeefULL);
    BOOST_TEST_EQ(gid0.get_msb(), 0x0ULL);

    BOOST_TEST_EQ(gid1.get_lsb(), 0xdeadbeefULL);
    BOOST_TEST_EQ(gid1.get_msb(), 0xcededeedULL);
   
    // assignment tests
    gid1 = 0xfeedbeefULL; // lsb assignment

    BOOST_TEST_EQ(gid1.get_lsb(), 0xfeedbeefULL);
    BOOST_TEST_EQ(gid1.get_msb(), 0x0ULL);
    
    { // basic increment tests
      gid_type gid(~0x0ULL);
    
      BOOST_TEST_EQ(gid1.get_msb(), 0x0ULL);
      BOOST_TEST_EQ((gid1++).get_lsb(), ~0x0ULL);
      BOOST_TEST_EQ((gid1++).get_msb(), 0x1ULL);
      BOOST_TEST_EQ(gid1.get_msb(), 0x0ULL);
      BOOST_TEST_EQ((++gid1).get_lsb(), ~0x0ULL);
      BOOST_TEST_EQ(gid1.get_msb(), 0x1ULL);
    }

    { // logical shift tests
      gid_type gid(~0x0ULL, ~0x0ULL);
      BOOST_TEST_EQ(gid.get_lsb(), 0); 
      BOOST_TEST_EQ(gid.get_msb(), 0); 
    }

    return boost::report_errors();
} 
