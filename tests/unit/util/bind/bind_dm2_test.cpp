#include <hpx/hpx_init.hpp>

#if defined(HPX_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_dm2_test.cpp - data members, advanced uses
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
#include <string>

#include <hpx/util/lightweight_test.hpp>

struct X
{
    int m;
};

struct Y
{
    char m[ 64 ];
};

int main()
{
    X x = { 0 };
    X * px = &x;

    hpx::util::bind( &X::m, placeholders::_1 )( px ) = 42;

    HPX_TEST( x.m == 42 );

    hpx::util::bind( &X::m, boost::ref(x) )() = 17041;

    HPX_TEST( x.m == 17041 );

    X const * pcx = &x;

    HPX_TEST( hpx::util::bind( &X::m, placeholders::_1 )( pcx ) == 17041L );
    HPX_TEST( hpx::util::bind( &X::m, pcx )() == 17041L );

    Y y = { "test" };
    std::string v( "test" );

    HPX_TEST( hpx::util::bind( &Y::m, &y )() == v );
    HPX_TEST( hpx::util::bind( &Y::m, &y )() == v );

    return hpx::util::report_errors();
}
