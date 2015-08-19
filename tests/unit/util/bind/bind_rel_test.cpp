#include <hpx/hpx_init.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_rel_test.cpp - ==, !=, <, <=, >, >= operators
//
//  Copyright (c) 2005 Peter Dimov
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/util/bind.hpp>

namespace placeholders = hpx::util::placeholders;

#include <iostream>

#include <hpx/util/lightweight_test.hpp>

int f( int x )
{
    return x + x;
}

int g( int x )
{
    return 2 * x;
}

int main()
{
    int x = 4;
    int y = x + x;

    // bind op value

    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) == y )( x ) );
    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) != y )( x ) ) );

    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) < y )( x ) ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) < y + 1 )( x ) );

    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) > y )( x ) ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) > y - 1 )( x ) );

    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) <= y - 1 )( x ) ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) <= y )( x ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) <= y + 1 )( x ) );

    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) >= y + 1 )( x ) ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) >= y )( x ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) >= y - 1 )( x ) );

    // bind op ref

    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) == boost::ref( y ) )( x ) );
    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) != boost::ref( y ) )( x ) ) );
    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) < boost::ref( y ) )( x ) ) );
    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) > boost::ref( y ) )( x ) ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) <= boost::ref( y ) )( x ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) >= boost::ref( y ) )( x ) );

    // bind op placeholder

    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) == placeholders::_2 )( x, y ) );
    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) !=
        placeholders::_2 )( x, y ) ) );
    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) <
        placeholders::_2 )( x, y ) ) );
    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) >
        placeholders::_2 )( x, y ) ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) <= placeholders::_2 )( x, y ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) >= placeholders::_2 )( x, y ) );

    // bind op bind

    // important: hpx::util::bind( f, placeholders::_1 ) and
    // hpx::util::bind( g, placeholders::_1 ) have the same type
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) ==
        hpx::util::bind( g, placeholders::_1 ) )( x ) );
    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) !=
        hpx::util::bind( g, placeholders::_1 ) )( x ) ) );
    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) <
        hpx::util::bind( g, placeholders::_1 ) )( x ) ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) <=
        hpx::util::bind( g, placeholders::_1 ) )( x ) );
    HPX_TEST( !( ( hpx::util::bind( f, placeholders::_1 ) >
        hpx::util::bind( g, placeholders::_1 ) )( x ) ) );
    HPX_TEST( ( hpx::util::bind( f, placeholders::_1 ) >=
        hpx::util::bind( g, placeholders::_1 ) )( x ) );

    return hpx::util::report_errors();
}
