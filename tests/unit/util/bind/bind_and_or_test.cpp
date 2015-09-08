#include <hpx/hpx_init.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_and_or_test.cpp - &&, || operators
//
//  Copyright (c) 2008 Peter Dimov
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

bool f( bool x )
{
    return x;
}

bool g( bool x )
{
    return !x;
}

bool h()
{
    BOOST_ERROR( "Short-circuit evaluation failure" );
    return false;
}

template< class F, class A1, class A2, class R > void test( F f, A1 a1, A2 a2, R r )
{
    HPX_TEST( f( a1, a2 ) == r );
}

int main()
{
    // &&

    test( hpx::util::bind( f, true ) && hpx::util::bind( g, true ),
        false, false, f( true ) && g( true ) );
    test( hpx::util::bind( f, true ) && hpx::util::bind( g, false ),
        false, false, f( true ) && g( false ) );

    test( hpx::util::bind( f, false ) && hpx::util::bind( h ),
        false, false, f( false ) && h() );

    test( hpx::util::bind( f, placeholders::_1 ) && hpx::util::bind( g,
        placeholders::_2 ), true, true, f( true ) && g( true ) );
    test( hpx::util::bind( f, placeholders::_1 ) && hpx::util::bind( g,
        placeholders::_2 ), true, false, f( true ) && g( false ) );

    test( hpx::util::bind( f, placeholders::_1 ) && hpx::util::bind( h ),
        false, false, f( false ) && h() );

    // ||

    test( hpx::util::bind( f, false ) || hpx::util::bind( g, true ),
        false, false, f( false ) || g( true ) );
    test( hpx::util::bind( f, false ) || hpx::util::bind( g, false ),
        false, false, f( false ) || g( false ) );

    test( hpx::util::bind( f, true ) || hpx::util::bind( h ),
        false, false, f( true ) || h() );

    test( hpx::util::bind( f, placeholders::_1 ) || hpx::util::bind( g,
        placeholders::_2 ), false, true, f( false ) || g( true ) );
    test( hpx::util::bind( f, placeholders::_1 ) || hpx::util::bind( g,
        placeholders::_2 ), false, false, f( false ) || g( false ) );

    test( hpx::util::bind( f, placeholders::_1 ) || hpx::util::bind( h ),
        true, false, f( true ) || h() );

    //

    return hpx::util::report_errors();
}
