#include <hpx/hpx_init.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_dm_test.cpp - data members
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

struct X
{
    int m;
};

X f( int v )
{
    X r = { v };
    return r;
}

int main()
{
    X x = { 17041 };
    X * px = &x;

    HPX_TEST( hpx::util::bind( &X::m, placeholders::_1 )( x ) == 17041 );
    HPX_TEST( hpx::util::bind( &X::m, placeholders::_1 )( px ) == 17041 );

    HPX_TEST( hpx::util::bind( &X::m, x )() == 17041 );
    HPX_TEST( hpx::util::bind( &X::m, px )() == 17041 );
    HPX_TEST( hpx::util::bind( &X::m, boost::ref(x) )() == 17041 );


    X const cx = x;
    X const * pcx = &cx;

    HPX_TEST( hpx::util::bind( &X::m, placeholders::_1 )( cx ) == 17041 );
    HPX_TEST( hpx::util::bind( &X::m, placeholders::_1 )( pcx ) == 17041 );

    HPX_TEST( hpx::util::bind( &X::m, cx )() == 17041 );
    HPX_TEST( hpx::util::bind( &X::m, pcx )() == 17041 );
    HPX_TEST( hpx::util::bind( &X::m, boost::ref(cx) )() == 17041 );

    int const v = 42;

    HPX_TEST( hpx::util::bind( &X::m, hpx::util::bind( f,
        placeholders::_1 ) )( v ) == v );

    return hpx::util::report_errors();
}
