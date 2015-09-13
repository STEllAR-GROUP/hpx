#include <hpx/hpx_init.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_fastcall_test.cpp - test for bind.hpp + __fastcall (free functions)
//
//  Copyright (c) 2002 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#define BOOST_BIND_ENABLE_FASTCALL

#include <hpx/util/bind.hpp>

namespace placeholders = hpx::util::placeholders;

#include <iostream>

#include <hpx/util/lightweight_test.hpp>

//

long __fastcall f_0()
{
    return 17041L;
}

long __fastcall f_1(long a)
{
    return a;
}

long __fastcall f_2(long a, long b)
{
    return a + 10 * b;
}

long __fastcall f_3(long a, long b, long c)
{
    return a + 10 * b + 100 * c;
}

long __fastcall f_4(long a, long b, long c, long d)
{
    return a + 10 * b + 100 * c + 1000 * d;
}

long __fastcall f_5(long a, long b, long c, long d, long e)
{
    return a + 10 * b + 100 * c + 1000 * d + 10000 * e;
}

long __fastcall f_6(long a, long b, long c, long d, long e, long f)
{
    return a + 10 * b + 100 * c + 1000 * d + 10000 * e + 100000 * f;
}

long __fastcall f_7(long a, long b, long c, long d, long e, long f, long g)
{
    return a + 10 * b + 100 * c + 1000 * d + 10000 * e + 100000 * f + 1000000 * g;
}

long __fastcall f_8(long a, long b, long c, long d, long e, long f, long g, long h)
{
    return a + 10 * b + 100 * c + 1000 * d + 10000 * e
        + 100000 * f + 1000000 * g + 10000000 * h;
}

long __fastcall f_9(long a, long b, long c, long d, long e,
    long f, long g, long h, long i)
{
    return a + 10 * b + 100 * c + 1000 * d + 10000 * e
        + 100000 * f + 1000000 * g + 10000000 * h + 100000000 * i;
}

void function_test()
{
    using namespace boost;

    int const i = 1;

    HPX_TEST( hpx::util::bind(f_0)(i) == 17041L );
    HPX_TEST( hpx::util::bind(f_1, placeholders::_1)(i) == 1L );
    HPX_TEST( hpx::util::bind(f_2, placeholders::_1, 2)(i) == 21L );
    HPX_TEST( hpx::util::bind(f_3, placeholders::_1, 2, 3)(i) == 321L );
    HPX_TEST( hpx::util::bind(f_4, placeholders::_1, 2, 3, 4)(i) == 4321L );
    HPX_TEST( hpx::util::bind(f_5, placeholders::_1, 2, 3, 4, 5)(i) == 54321L );
    HPX_TEST( hpx::util::bind(f_6, placeholders::_1, 2, 3, 4, 5, 6)(i) == 654321L );
    HPX_TEST( hpx::util::bind(f_7, placeholders::_1, 2, 3, 4, 5,
        6, 7)(i) == 7654321L );
    HPX_TEST( hpx::util::bind(f_8, placeholders::_1, 2, 3, 4, 5,
        6, 7, 8)(i) == 87654321L );
    HPX_TEST( hpx::util::bind(f_9, placeholders::_1, 2, 3, 4, 5,
        6, 7, 8, 9)(i) == 987654321L );
}

int main()
{
    function_test();
    return hpx::util::report_errors();
}
