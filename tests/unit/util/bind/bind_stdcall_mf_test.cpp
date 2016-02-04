#include <hpx/hpx_init.hpp>

#if defined(HPX_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_stdcall_mf_test.cpp - test for bind.hpp + __stdcall (member functions)
//
//  Copyright (c) 2001 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#define BOOST_MEM_FN_ENABLE_STDCALL

#include <hpx/util/bind.hpp>

namespace placeholders = hpx::util::placeholders;

#include <iostream>

#include <hpx/util/lightweight_test.hpp>

struct X
{
    mutable unsigned int hash;

    X(): hash(0) {}

    int __stdcall f0() { f1(17); return 0; }
    int __stdcall g0() const { g1(17); return 0; }

    int __stdcall f1(int a1) { hash = (hash * 17041 + a1) % 32768; return 0; }
    int __stdcall g1(int a1) const { hash = (hash * 17041 + a1 * 2) % 32768; return 0; }

    int __stdcall f2(int a1, int a2) { f1(a1); f1(a2); return 0; }
    int __stdcall g2(int a1, int a2) const { g1(a1); g1(a2); return 0; }

    int __stdcall f3(int a1, int a2, int a3) { f2(a1, a2); f1(a3); return 0; }
    int __stdcall g3(int a1, int a2, int a3) const { g2(a1, a2); g1(a3); return 0; }

    int __stdcall f4(int a1, int a2, int a3, int a4)
        { f3(a1, a2, a3); f1(a4); return 0; }
    int __stdcall g4(int a1, int a2, int a3, int a4)
        const { g3(a1, a2, a3); g1(a4); return 0; }

    int __stdcall f5(int a1, int a2, int a3, int a4, int a5)
        { f4(a1, a2, a3, a4); f1(a5); return 0; }
    int __stdcall g5(int a1, int a2, int a3, int a4, int a5)
        const { g4(a1, a2, a3, a4); g1(a5); return 0; }

    int __stdcall f6(int a1, int a2, int a3, int a4, int a5, int a6)
        { f5(a1, a2, a3, a4, a5); f1(a6); return 0; }
    int __stdcall g6(int a1, int a2, int a3, int a4, int a5, int a6)
        const { g5(a1, a2, a3, a4, a5); g1(a6); return 0; }

    int __stdcall f7(int a1, int a2, int a3, int a4, int a5, int a6, int a7)
        { f6(a1, a2, a3, a4, a5, a6); f1(a7); return 0; }
    int __stdcall g7(int a1, int a2, int a3, int a4, int a5, int a6, int a7)
        const { g6(a1, a2, a3, a4, a5, a6); g1(a7); return 0; }

    int __stdcall f8(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8)
        { f7(a1, a2, a3, a4, a5, a6, a7); f1(a8); return 0; }
    int __stdcall g8(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8)
        const { g7(a1, a2, a3, a4, a5, a6, a7); g1(a8); return 0; }
};

void member_function_test()
{
    using namespace boost;

    X x;

    // 0

    hpx::util::bind(&X::f0, &x)();
    hpx::util::bind(&X::f0, ref(x))();

    hpx::util::bind(&X::g0, &x)();
    hpx::util::bind(&X::g0, x)();
    hpx::util::bind(&X::g0, ref(x))();

    // 1

    hpx::util::bind(&X::f1, &x, 1)();
    hpx::util::bind(&X::f1, ref(x), 1)();

    hpx::util::bind(&X::g1, &x, 1)();
    hpx::util::bind(&X::g1, x, 1)();
    hpx::util::bind(&X::g1, ref(x), 1)();

    // 2

    hpx::util::bind(&X::f2, &x, 1, 2)();
    hpx::util::bind(&X::f2, ref(x), 1, 2)();

    hpx::util::bind(&X::g2, &x, 1, 2)();
    hpx::util::bind(&X::g2, x, 1, 2)();
    hpx::util::bind(&X::g2, ref(x), 1, 2)();

    // 3

    hpx::util::bind(&X::f3, &x, 1, 2, 3)();
    hpx::util::bind(&X::f3, ref(x), 1, 2, 3)();

    hpx::util::bind(&X::g3, &x, 1, 2, 3)();
    hpx::util::bind(&X::g3, x, 1, 2, 3)();
    hpx::util::bind(&X::g3, ref(x), 1, 2, 3)();

    // 4

    hpx::util::bind(&X::f4, &x, 1, 2, 3, 4)();
    hpx::util::bind(&X::f4, ref(x), 1, 2, 3, 4)();

    hpx::util::bind(&X::g4, &x, 1, 2, 3, 4)();
    hpx::util::bind(&X::g4, x, 1, 2, 3, 4)();
    hpx::util::bind(&X::g4, ref(x), 1, 2, 3, 4)();

    // 5

    hpx::util::bind(&X::f5, &x, 1, 2, 3, 4, 5)();
    hpx::util::bind(&X::f5, ref(x), 1, 2, 3, 4, 5)();

    hpx::util::bind(&X::g5, &x, 1, 2, 3, 4, 5)();
    hpx::util::bind(&X::g5, x, 1, 2, 3, 4, 5)();
    hpx::util::bind(&X::g5, ref(x), 1, 2, 3, 4, 5)();

    // 6

    hpx::util::bind(&X::f6, &x, 1, 2, 3, 4, 5, 6)();
    hpx::util::bind(&X::f6, ref(x), 1, 2, 3, 4, 5, 6)();

    hpx::util::bind(&X::g6, &x, 1, 2, 3, 4, 5, 6)();
    hpx::util::bind(&X::g6, x, 1, 2, 3, 4, 5, 6)();
    hpx::util::bind(&X::g6, ref(x), 1, 2, 3, 4, 5, 6)();

    // 7

    hpx::util::bind(&X::f7, &x, 1, 2, 3, 4, 5, 6, 7)();
    hpx::util::bind(&X::f7, ref(x), 1, 2, 3, 4, 5, 6, 7)();

    hpx::util::bind(&X::g7, &x, 1, 2, 3, 4, 5, 6, 7)();
    hpx::util::bind(&X::g7, x, 1, 2, 3, 4, 5, 6, 7)();
    hpx::util::bind(&X::g7, ref(x), 1, 2, 3, 4, 5, 6, 7)();

    // 8

    hpx::util::bind(&X::f8, &x, 1, 2, 3, 4, 5, 6, 7, 8)();
    hpx::util::bind(&X::f8, ref(x), 1, 2, 3, 4, 5, 6, 7, 8)();

    hpx::util::bind(&X::g8, &x, 1, 2, 3, 4, 5, 6, 7, 8)();
    hpx::util::bind(&X::g8, x, 1, 2, 3, 4, 5, 6, 7, 8)();
    hpx::util::bind(&X::g8, ref(x), 1, 2, 3, 4, 5, 6, 7, 8)();

    HPX_TEST( x.hash == 23558 );
}

int main()
{
    member_function_test();
    return hpx::util::report_errors();
}
