#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>

#if defined(HPX_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

// Taken from the Boost.Bind library
//
//  mem_fn_dm_test.cpp - data members
//
//  Copyright (c) 2005 Peter Dimov
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/util/mem_fn.hpp>

#include <iostream>

#include <hpx/util/lightweight_test.hpp>

struct X
{
    int m;
};

int main()
{
    X x = { 0 };

    hpx::util::mem_fn( &X::m )( x ) = 401;

    HPX_TEST( x.m == 401 );
    HPX_TEST( hpx::util::mem_fn( &X::m )( x ) == 401 );

    hpx::util::mem_fn( &X::m )( &x ) = 502;

    HPX_TEST( x.m == 502 );
    HPX_TEST( hpx::util::mem_fn( &X::m )( &x ) == 502 );

    X * px = &x;

    hpx::util::mem_fn( &X::m )( px ) = 603;

    HPX_TEST( x.m == 603 );
    HPX_TEST( hpx::util::mem_fn( &X::m )( px ) == 603 );

    X const & cx = x;
    X const * pcx = &x;

    HPX_TEST( hpx::util::mem_fn( &X::m )( cx ) == 603 );
    HPX_TEST( hpx::util::mem_fn( &X::m )( pcx ) == 603 );

    return hpx::util::report_errors();
}
