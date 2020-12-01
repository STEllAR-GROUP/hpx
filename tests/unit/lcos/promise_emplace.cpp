//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/testing.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void test_nullary_void()
{
#if !defined(HPX_COMPUTE_DEVICE_CODE)
    {
        hpx::lcos::promise<void> p;
        hpx::future<void> f = p.get_future();
        p.set_value();
        HPX_TEST(f.is_ready());
    }
#endif

    {
        hpx::lcos::local::promise<void> p;
        hpx::future<void> f = p.get_future();
        p.set_value();
        HPX_TEST(f.is_ready());
    }
}

struct A
{
    A() : i_(42) {}
    A(int i) : i_(i) {}

    int i_;
};

void test_nullary()
{
#if !defined(HPX_COMPUTE_DEVICE_CODE)
    {
        hpx::lcos::promise<A> p;
        hpx::future<A> f = p.get_future();
        p.set_value();
        HPX_TEST(f.is_ready());
        HPX_TEST_EQ(f.get().i_, 42);
    }
#endif

    {
        hpx::lcos::local::promise<A> p;
        hpx::future<A> f = p.get_future();
        p.set_value();
        HPX_TEST(f.is_ready());
        HPX_TEST_EQ(f.get().i_, 42);
    }
}

void test_unary()
{
#if !defined(HPX_COMPUTE_DEVICE_CODE)
    {
        hpx::lcos::promise<A> p1;
        hpx::future<A> f1 = p1.get_future();
        p1.set_value(A(42));
        HPX_TEST(f1.is_ready());
        HPX_TEST_EQ(f1.get().i_, 42);

        hpx::lcos::promise<A> p2;
        hpx::future<A> f2 = p2.get_future();
        p2.set_value(42);
        HPX_TEST(f2.is_ready());
        HPX_TEST_EQ(f2.get().i_, 42);
    }
#endif

    {
        hpx::lcos::local::promise<A> p1;
        hpx::future<A> f1 = p1.get_future();
        p1.set_value(A(42));
        HPX_TEST(f1.is_ready());
        HPX_TEST_EQ(f1.get().i_, 42);

        hpx::lcos::local::promise<A> p2;
        hpx::future<A> f2 = p2.get_future();
        p2.set_value(42);
        HPX_TEST(f2.is_ready());
        HPX_TEST_EQ(f2.get().i_, 42);
    }
}

struct B
{
    B(int i) : i_(i), j_(0) {}
    B(int i, int j) : i_(i), j_(j) {}

    B(B const&) = delete;
    B(B&& rhs) : i_(rhs.i_), j_(rhs.j_) {}

    int i_;
    int j_;
};

void test_variadic()
{
#if !defined(HPX_COMPUTE_DEVICE_CODE)
    {
        hpx::lcos::promise<B> p1;
        hpx::future<B> f1 = p1.get_future();
        p1.set_value(B(42));
        HPX_TEST(f1.is_ready());
        B r1 = f1.get();
        HPX_TEST_EQ(r1.i_, 42);
        HPX_TEST_EQ(r1.j_, 0);

        hpx::lcos::promise<B> p2;
        hpx::future<B> f2 = p2.get_future();
        p2.set_value(42);
        HPX_TEST(f2.is_ready());
        B r2 = f2.get();
        HPX_TEST_EQ(r2.i_, 42);
        HPX_TEST_EQ(r2.j_, 0);

        hpx::lcos::promise<B> p3;
        hpx::future<B> f3 = p3.get_future();
        p3.set_value(B(42, 43));
        HPX_TEST(f3.is_ready());
        B r3 = f3.get();
        HPX_TEST_EQ(r3.i_, 42);
        HPX_TEST_EQ(r3.j_, 43);

        hpx::lcos::promise<B> p4;
        hpx::future<B> f4 = p4.get_future();
        p4.set_value(42, 43);
        HPX_TEST(f4.is_ready());
        B r4 = f4.get();
        HPX_TEST_EQ(r4.i_, 42);
        HPX_TEST_EQ(r4.j_, 43);
    }
#endif

    {
        hpx::lcos::local::promise<B> p1;
        hpx::future<B> f1 = p1.get_future();
        p1.set_value(B(42));
        HPX_TEST(f1.is_ready());
        B r1 = f1.get();
        HPX_TEST_EQ(r1.i_, 42);
        HPX_TEST_EQ(r1.j_, 0);

        hpx::lcos::local::promise<B> p2;
        hpx::future<B> f2 = p2.get_future();
        p2.set_value(42);
        HPX_TEST(f2.is_ready());
        B r2 = f2.get();
        HPX_TEST_EQ(r2.i_, 42);
        HPX_TEST_EQ(r2.j_, 0);

        hpx::lcos::local::promise<B> p3;
        hpx::future<B> f3 = p3.get_future();
        p3.set_value(B(42, 43));
        HPX_TEST(f3.is_ready());
        B r3 = f3.get();
        HPX_TEST_EQ(r3.i_, 42);
        HPX_TEST_EQ(r3.j_, 43);

        hpx::lcos::local::promise<B> p4;
        hpx::future<B> f4 = p4.get_future();
        p4.set_value(42, 43);
        HPX_TEST(f4.is_ready());
        B r4 = f4.get();
        HPX_TEST_EQ(r4.i_, 42);
        HPX_TEST_EQ(r4.j_, 43);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_nullary_void();
    test_nullary();

    test_unary();
    test_variadic();

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return hpx::util::report_errors();
}

