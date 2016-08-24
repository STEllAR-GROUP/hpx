//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
std::int32_t increment(std::int32_t i)
{
    return i + 1;
}

std::int32_t increment_with_future(hpx::shared_future<std::int32_t> fi)
{
    return fi.get() + 1;
}

///////////////////////////////////////////////////////////////////////////////
struct mult2
{
    std::int32_t operator()(std::int32_t i) const
    {
        return i * 2;
    }
};

///////////////////////////////////////////////////////////////////////////////
struct decrement
{
    std::int32_t call(std::int32_t i) const
    {
        return i - 1;
    }
};

///////////////////////////////////////////////////////////////////////////////
void do_nothing(std::int32_t i)
{
}

struct do_nothing_obj
{
    void operator()(std::int32_t i) const
    {
    }
};

struct do_nothing_member
{
    void call(std::int32_t i) const
    {
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        hpx::future<std::int32_t> f1 = hpx::async(&increment, 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<std::int32_t> f2 =
            hpx::async(hpx::launch::all, &increment, 42);
        HPX_TEST_EQ(f2.get(), 43);

        hpx::future<void> f3 = hpx::async(&do_nothing, 42);
        f3.get();

        hpx::future<void> f4 = hpx::async(hpx::launch::sync, &do_nothing, 42);
        f4.get();
    }

    {
        hpx::promise<std::int32_t> p;
        hpx::shared_future<std::int32_t> f = p.get_future();

        hpx::future<std::int32_t> f1 = hpx::async(&increment_with_future, f);
        hpx::future<std::int32_t> f2 =
            hpx::async(hpx::launch::all, &increment_with_future, f);

        p.set_value(42);
        HPX_TEST_EQ(f1.get(), 43);
        HPX_TEST_EQ(f2.get(), 43);
    }

    {
        using hpx::util::placeholders::_1;

        hpx::future<std::int32_t> f1 =
            hpx::async(hpx::util::bind(&increment, 42));
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<std::int32_t> f2 =
            hpx::async(hpx::launch::all, hpx::util::bind(&increment, _1), 42);
        HPX_TEST_EQ(f2.get(), 43);

        hpx::future<std::int32_t> f3 =
            hpx::async(hpx::util::bind(&increment, 42));
        HPX_TEST_EQ(f3.get(), 43);

        hpx::future<std::int32_t> f4 =
            hpx::async(hpx::launch::all, hpx::util::bind(&increment, _1), 42);
        HPX_TEST_EQ(f4.get(), 43);

        hpx::future<void> f5 = hpx::async(hpx::launch::all,
            hpx::util::bind(&do_nothing, _1), 42);
        f5.get();

        hpx::future<void> f6 = hpx::async(hpx::launch::sync,
            hpx::util::bind(&do_nothing, _1), 42);
        f6.get();
    }

    {
        hpx::future<std::int32_t> f1 = hpx::async(increment, 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<std::int32_t> f2 =
            hpx::async(hpx::launch::all, increment, 42);
        HPX_TEST_EQ(f2.get(), 43);

        hpx::future<void> f3 = hpx::async(do_nothing, 42);
        f3.get();

        hpx::future<void> f4 = hpx::async(hpx::launch::sync, do_nothing, 42);
        f4.get();
    }

    {
        using hpx::util::placeholders::_1;

        hpx::future<std::int32_t> f1 =
            hpx::async(hpx::util::bind(increment, 42));
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<std::int32_t> f2 =
            hpx::async(hpx::launch::all, hpx::util::bind(increment, _1), 42);
        HPX_TEST_EQ(f2.get(), 43);

        hpx::future<std::int32_t> f3 =
            hpx::async(hpx::util::bind(increment, 42));
        HPX_TEST_EQ(f3.get(), 43);

        hpx::future<std::int32_t> f4 =
            hpx::async(hpx::launch::all, hpx::util::bind(increment, _1), 42);
        HPX_TEST_EQ(f4.get(), 43);

        hpx::future<void> f5 = hpx::async(hpx::launch::all,
            hpx::util::bind(do_nothing, _1), 42);
        f5.get();

        hpx::future<void> f6 = hpx::async(hpx::launch::sync,
            hpx::util::bind(do_nothing, _1), 42);
        f6.get();
    }

    {
        mult2 mult;

        hpx::future<std::int32_t> f1 = hpx::async(mult, 42);
        HPX_TEST_EQ(f1.get(), 84);

        hpx::future<std::int32_t> f2 = hpx::async(hpx::launch::all, mult, 42);
        HPX_TEST_EQ(f2.get(), 84);
    }

    {
        mult2 mult;

        hpx::future<std::int32_t> f1 =
           hpx::async(hpx::util::bind(mult, 42));
        HPX_TEST_EQ(f1.get(), 84);

        using hpx::util::placeholders::_1;

        hpx::future<std::int32_t> f2 =
            hpx::async(hpx::launch::all, hpx::util::bind(mult, 42));
        HPX_TEST_EQ(f2.get(), 84);

        hpx::future<std::int32_t> f3 =
           hpx::async(hpx::util::bind(mult, _1), 42);
        HPX_TEST_EQ(f3.get(), 84);

        hpx::future<std::int32_t> f4 =
            hpx::async(hpx::launch::all, hpx::util::bind(mult, _1), 42);
        HPX_TEST_EQ(f4.get(), 84);

        do_nothing_obj do_nothing_f;
        hpx::future<void> f5 = hpx::async(hpx::launch::all,
            hpx::util::bind(do_nothing_f, _1), 42);
        f5.get();

        hpx::future<void> f6 = hpx::async(hpx::launch::sync,
            hpx::util::bind(do_nothing_f, _1), 42);
        f6.get();
    }

    {
        decrement dec;

        hpx::future<std::int32_t> f1 = hpx::async(&decrement::call, dec, 42);
        HPX_TEST_EQ(f1.get(), 41);

        hpx::future<std::int32_t> f2 =
            hpx::async(hpx::launch::all, &decrement::call, dec, 42);
        HPX_TEST_EQ(f2.get(), 41);

        do_nothing_member dnm;
        hpx::future<void> f3 = hpx::async(hpx::launch::all,
            &do_nothing_member::call, dnm, 42);
        f3.get();

        hpx::future<void> f4 = hpx::async(hpx::launch::sync,
            &do_nothing_member::call, dnm, 42);
        f4.get();
    }

    {
        decrement dec;

        using hpx::util::placeholders::_1;

        hpx::future<std::int32_t> f1 =
            hpx::async(hpx::util::bind(&decrement::call, dec, 42));
        HPX_TEST_EQ(f1.get(), 41);

        hpx::future<std::int32_t> f2 = hpx::async(
            hpx::launch::all, hpx::util::bind(&decrement::call, dec, 42));
        HPX_TEST_EQ(f2.get(), 41);

        hpx::future<std::int32_t> f3 =
            hpx::async(hpx::util::bind(&decrement::call, dec, _1), 42);
        HPX_TEST_EQ(f3.get(), 41);

        hpx::future<std::int32_t> f4 = hpx::async(
            hpx::launch::all, hpx::util::bind(&decrement::call, dec, _1), 42);
        HPX_TEST_EQ(f4.get(), 41);

        do_nothing_member dnm;
        hpx::future<void> f5 = hpx::async(hpx::launch::all,
            hpx::util::bind(&do_nothing_member::call, dnm, _1), 42);
        f5.get();

        hpx::future<void> f6 = hpx::async(hpx::launch::sync,
            hpx::util::bind(&do_nothing_member::call, dnm, _1), 42);
        f6.get();
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

