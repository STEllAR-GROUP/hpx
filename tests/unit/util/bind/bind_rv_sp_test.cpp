#include <hpx/hpx_init.hpp>

#if defined(HPX_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_rv_sp_test.cpp - smart pointer returned by value from an inner bind
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
#include <boost/shared_ptr.hpp>

struct X
{
    int v_;

    X( int v ): v_( v )
    {
    }

    int f()
    {
        return v_;
    }
};

struct Y
{
    boost::shared_ptr<X> f()
    {
        return boost::shared_ptr<X>( new X( 42 ) );
    }
};

int main()
{
    Y y;

    HPX_TEST( hpx::util::bind( &X::f, hpx::util::bind( &Y::f, &y ) )() == 42 );

    return hpx::util::report_errors();
}
