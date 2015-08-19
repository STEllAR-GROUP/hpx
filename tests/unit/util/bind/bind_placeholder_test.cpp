#include <hpx/hpx_init.hpp>

#if defined( BOOST_MSVC )

#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed

#endif

//  Taken from the Boost.Bind library
//
//  bind_placeholder_test.cpp - test custom placeholders
//
//  Copyright (c) 2006 Peter Dimov
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0.
//
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/bind.hpp>

namespace placeholders = hpx::util::placeholders;

#include <iostream>

#include <hpx/util/lightweight_test.hpp>

//

long f( long a, long b, long c, long d, long e, long f, long g, long h, long i )
{
    return a + 10 * b + 100 * c + 1000 * d + 10000 * e
        + 100000 * f + 1000000 * g + 10000000 * h + 100000000 * i;
}

template< int I > struct custom_placeholder
{
};

namespace hpx { namespace traits
{

template< int I > struct is_placeholder< custom_placeholder< I > >
{
    enum { value = I };
};

}} // namespace boost

int main()
{
    int const x1 = 1;
    int const x2 = 2;
    int const x3 = 3;
    int const x4 = 4;
    int const x5 = 5;
    int const x6 = 6;
    int const x7 = 7;
    int const x8 = 8;
    int const x9 = 9;

    custom_placeholder<1> p1;
    custom_placeholder<2> p2;
    custom_placeholder<3> p3;
    custom_placeholder<4> p4;
    custom_placeholder<5> p5;
    custom_placeholder<6> p6;
    custom_placeholder<7> p7;
    custom_placeholder<8> p8;
    custom_placeholder<9> p9;

    HPX_TEST(
        hpx::util::bind( f, p1, p2, p3, p4, p5, p6, p7, p8, p9 )
        ( x1, x2, x3, x4, x5, x6, x7, x8, x9 ) == 987654321L );

    return hpx::util::report_errors();
}
