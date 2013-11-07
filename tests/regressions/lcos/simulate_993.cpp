//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test tries to reproduce part of the code we use in HPX to allow for promises
// being managed as type erased entities. The resulting code is close to UB. This
// test should help verifying the trick is usable on all platforms.

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstdlib>
#include <cstring>

// base_lco_with_value
struct A
{
    virtual void a1() {}
    virtual void a2() {}
};

// future_data
struct B_base
{
    virtual void b1() = 0;
    virtual void b2() = 0;
};

struct B1 : B_base
{
    virtual void b1() {}
    virtual void b2() {}
};

struct B2 : B_base
{
    B2()
    {
        std::memset(buffer, 0, sizeof(buffer));
    }

    ~B2()
    {
        char b[16] = { 0 };
        HPX_TEST(0 == std::memcmp(buffer, b, sizeof(buffer)));
    }

    virtual void b1() {}
    virtual void b2() {}

    char buffer[16];
};

// detail::promise
struct C1 : A, B1
{
    C1() : c(0) {}

    virtual void c1()
    {
        c = ~0x0;
    }

    void c2()
    {
        c1();
    }

    int c;
};

struct C2 : A, B2
{
    C2() : c(0) {}

    ~C2()
    {
        HPX_TEST_EQ(~0x0, c);
    }

    virtual void c1()
    {
        c = ~0x0;
    }

    void c2()
    {
        c1();
    }

    int c;
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        C2 cobj;
        void* base = &cobj;
        reinterpret_cast<C1*>(base)->c2();      // should invoke C2::c1()
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

int main(int argc, char** argv)
{
    return hpx::init(argc, argv);
}
