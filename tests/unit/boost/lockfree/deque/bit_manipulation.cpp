////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/cstdint.hpp>
#include <boost/lockfree/detail/bit_manipulation.hpp>

#include <hpx/util/lightweight_test.hpp>

int main()
{
    using boost::lockfree::detail::get_bit_range;
    using boost::lockfree::detail::pack_bits;

    boost::uint32_t abcd = 0xaabbccdd;

    boost::uint8_t d = get_bit_range<0, 8, boost::uint8_t>(abcd),
                   c = get_bit_range<8, 16, boost::uint8_t>(abcd),
                   b = get_bit_range<16, 24, boost::uint8_t>(abcd),
                   a = get_bit_range<24, 32, boost::uint8_t>(abcd); 

    HPX_TEST_EQ(d, 0xdd);
    HPX_TEST_EQ(c, 0xcc);
    HPX_TEST_EQ(b, 0xbb);
    HPX_TEST_EQ(a, 0xaa);

    boost::uint32_t rebuilt = pack_bits<0, boost::uint32_t>(d)
                            + pack_bits<8, boost::uint32_t>(c)
                            + pack_bits<16, boost::uint32_t>(b)
                            + pack_bits<24, boost::uint32_t>(a);

    HPX_TEST_EQ(rebuilt, abcd);

    return hpx::util::report_errors();
}

