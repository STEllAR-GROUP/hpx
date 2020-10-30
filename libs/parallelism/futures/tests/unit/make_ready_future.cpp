//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/testing.hpp>

#include <functional>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void test_nullary_void()
{
    hpx::future<void> f1 = hpx::make_ready_future();
    HPX_TEST(f1.is_ready());

    hpx::future<void> f2 = hpx::make_ready_future<void>();
    HPX_TEST(f2.is_ready());
}

struct A
{
    A() = default;
};

void test_nullary()
{
    hpx::future<A> f1 = hpx::make_ready_future<A>();
    HPX_TEST(f1.is_ready());
}

struct B
{
    B(int i)
      : i_(i)
    {
    }

    int i_;
};

void test_unary()
{
    B lval(42);

    hpx::future<B> f1 = hpx::make_ready_future(B(42));
    HPX_TEST(f1.is_ready());
    HPX_TEST_EQ(f1.get().i_, 42);

    hpx::future<B> f2 = hpx::make_ready_future(lval);
    HPX_TEST(f2.is_ready());
    HPX_TEST_EQ(f2.get().i_, 42);

    hpx::future<B> f3 = hpx::make_ready_future<B>(42);
    HPX_TEST(f3.is_ready());
    HPX_TEST_EQ(f3.get().i_, 42);

    hpx::future<B&> f4 = hpx::make_ready_future(std::ref(lval));
    HPX_TEST(f4.is_ready());
    HPX_TEST_EQ(&f4.get().i_, &lval.i_);

    hpx::future<B&> f5 = hpx::make_ready_future<B&>(lval);
    HPX_TEST(f5.is_ready());
    HPX_TEST_EQ(&f5.get().i_, &lval.i_);
}

struct C
{
    C(int i)
      : i_(i)
      , j_(0)
    {
    }
    C(int i, int j)
      : i_(i)
      , j_(j)
    {
    }

    C(C const&) = delete;
    C(C&& rhs)
      : i_(rhs.i_)
      , j_(rhs.j_)
    {
    }

    int i_;
    int j_;
};

void test_variadic()
{
    hpx::future<C> f1 = hpx::make_ready_future(C(42));
    HPX_TEST(f1.is_ready());
    C r1 = f1.get();
    HPX_TEST_EQ(r1.i_, 42);
    HPX_TEST_EQ(r1.j_, 0);

    hpx::future<C> f2 = hpx::make_ready_future<C>(42);
    HPX_TEST(f2.is_ready());
    C r2 = f2.get();
    HPX_TEST_EQ(r2.i_, 42);
    HPX_TEST_EQ(r2.j_, 0);

    hpx::future<C> f3 = hpx::make_ready_future(C(42, 43));
    HPX_TEST(f3.is_ready());
    C r3 = f3.get();
    HPX_TEST_EQ(r3.i_, 42);
    HPX_TEST_EQ(r3.j_, 43);

    hpx::future<C> f4 = hpx::make_ready_future<C>(42, 43);
    HPX_TEST(f4.is_ready());
    C r4 = f4.get();
    HPX_TEST_EQ(r4.i_, 42);
    HPX_TEST_EQ(r4.j_, 43);
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
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return hpx::util::report_errors();
}
