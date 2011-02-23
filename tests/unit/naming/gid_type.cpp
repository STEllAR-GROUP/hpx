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

    BOOST_TEST_EQ(gid0.get_msb(), 0x0ULL);
    BOOST_TEST_EQ(gid0.get_lsb(), 0xdeadbeefULL);

    BOOST_TEST_EQ(gid1.get_msb(), 0xdeadbeefULL);
    BOOST_TEST_EQ(gid1.get_lsb(), 0xcededeedULL);
   
    // assignment tests
    gid1 = 0xfeedbeefULL; // lsb assignment

    BOOST_TEST_EQ(gid1.get_msb(), 0x0ULL);
    BOOST_TEST_EQ(gid1.get_lsb(), 0xfeedbeefULL);
    
    { // post-increment tests (post-increment should return a temporary)
      gid_type gid0(~0x0ULL), // special-case test 
               gid1(0xdeULL); // control group for < ~0x0ULL
               gid2(0xdeULL, 0xadULL) // control group for > ~0x0ULL

      // 0x00000000ffffffff + 0x1 = 0x0000000100000000    
      BOOST_TEST_EQ((gid0++).get_msb(), 0x0ULL);
      BOOST_TEST_EQ((gid0++).get_lsb(), ~0x0ULL); 

      // 0x000000de00000000 + 0x1 = 0x000000de00000001    
      BOOST_TEST_EQ((gid1++).get_msb(), 0xdeULL);
      BOOST_TEST_EQ((gid1++).get_lsb(), 0x0ULL); 

      // 0x000000de000000ad + 0x1 = 0x000000de000000ae    
      BOOST_TEST_EQ((gid2++).get_msb(), 0xdeULL);
      BOOST_TEST_EQ((gid2++).get_lsb(), 0xadULL); 

      // check that the originals were modified 
      BOOST_TEST_EQ(gid0.get_msb(), 0x1ULL);
      BOOST_TEST_EQ(gid0.get_lsb(), 0x0ULL); 
      BOOST_TEST_EQ(gid1.get_msb(), 0xdeULL);
      BOOST_TEST_EQ(gid1.get_lsb(), 0x1ULL); 
      BOOST_TEST_EQ(gid2.get_msb(), 0xdeULL);
      BOOST_TEST_EQ(gid2.get_lsb(), 0xaeULL); 
    }
    
    { // pre-increment tests (post-increment should return the original)
      gid_type gid0(~0x0ULL), // special-case test 
               gid1(0xdeULL); // control group for < ~0x0ULL
               gid2(0xdeULL, 0xadULL) // control group for > ~0x0ULL

      // 0x00000000ffffffff + 0x1 = 0x0000000100000000    
      BOOST_TEST_EQ((++gid0).get_msb(), 0x1ULL);
      BOOST_TEST_EQ((++gid0).get_lsb(), 0x0ULL); 

      // 0x000000de00000000 + 0x1 = 0x000000de00000001    
      BOOST_TEST_EQ((++gid1).get_msb(), 0xdeULL);
      BOOST_TEST_EQ((++gid1).get_lsb(), 0x1ULL); 

      // 0x000000de000000ad + 0x1 = 0x000000de000000ae    
      BOOST_TEST_EQ((++gid2).get_msb(), 0xdeULL);
      BOOST_TEST_EQ((++gid2).get_lsb(), 0xaeULL); 

      // check that the originals were modified 
      BOOST_TEST_EQ(gid0.get_msb(), 0x1ULL);
      BOOST_TEST_EQ(gid0.get_lsb(), 0x0ULL); 
      BOOST_TEST_EQ(gid1.get_msb(), 0xdeULL);
      BOOST_TEST_EQ(gid1.get_lsb(), 0x1ULL); 
      BOOST_TEST_EQ(gid2.get_msb(), 0xdeULL);
      BOOST_TEST_EQ(gid2.get_lsb(), 0xaeULL); 
    }

    { // logical shift tests
      gid_type gid(~0x0ULL, ~0x0ULL);
      ++gid; // should cause a shift 
      // in C/C++, 0xffffffffffffffff + 0x1 = 0x0000000000000000    
      BOOST_TEST_EQ(gid.get_lsb(), 0); 
      BOOST_TEST_EQ(gid.get_msb(), 0); 
    }

    { // boolean conversions
      gid_type gid; // should be default initialized to 0
      BOOST_TEST_EQ(gid, false);
      gid = 12;
      BOOST_TEST_EQ(gid, true);
    }

    return boost::report_errors();
} 
