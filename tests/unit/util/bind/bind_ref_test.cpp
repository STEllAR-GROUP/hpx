//  Taken from the Boost.Bind library
//
//  bind_ref_test.cpp - reference_wrapper
//
//  Copyright (c) 2009 Peter Dimov
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt
//

#include <hpx/hpx_init.hpp>
#include <hpx/util/bind.hpp>
#include <boost/ref.hpp>
#include <hpx/util/lightweight_test.hpp>

namespace placeholders = hpx::util::placeholders;

struct X
{
    int f( int x )
    {
        return x;
    }

    int g( int x ) const
    {
        return -x;
    }
};

int main()
{
    X x;

    HPX_TEST( hpx::util::bind( &X::f, placeholders::_1, 1 )( boost::ref( x ) ) == 1 );
    HPX_TEST( hpx::util::bind( &X::g, placeholders::_1, 2 )( boost::cref( x ) ) == -2 );

    return hpx::util::report_errors();
}
