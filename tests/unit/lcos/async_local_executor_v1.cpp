//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/parallel_executors.hpp>
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
template <typename Executor>
void test_async_with_executor(Executor& exec)
{
    {
        hpx::future<std::int32_t> f1 = hpx::async(exec, &increment, 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<void> f2 = hpx::async(exec, &do_nothing, 42);
        f2.get();
    }

    {
        hpx::promise<std::int32_t> p;
        hpx::shared_future<std::int32_t> f = p.get_future();

        hpx::future<std::int32_t> f1 =
            hpx::async(exec, &increment_with_future, f);
        hpx::future<std::int32_t> f2 =
            hpx::async(exec, &increment_with_future, f);

        p.set_value(42);
        HPX_TEST_EQ(f1.get(), 43);
        HPX_TEST_EQ(f2.get(), 43);
    }

    {
        using hpx::util::placeholders::_1;

        hpx::future<std::int32_t> f1 =
            hpx::async(exec, hpx::util::bind(&increment, 42));
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<std::int32_t> f2 =
            hpx::async(exec, hpx::util::bind(&increment, _1), 42);
        HPX_TEST_EQ(f2.get(), 43);
    }

    {
        hpx::future<std::int32_t> f1 = hpx::async(exec, increment, 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<void> f2 = hpx::async(exec, do_nothing, 42);
        f2.get();
    }

    {
        mult2 mult;

        hpx::future<std::int32_t> f1 = hpx::async(exec, mult, 42);
        HPX_TEST_EQ(f1.get(), 84);
    }

    {
        mult2 mult;

        hpx::future<std::int32_t> f1 =
           hpx::async(exec, hpx::util::bind(mult, 42));
        HPX_TEST_EQ(f1.get(), 84);

        using hpx::util::placeholders::_1;

        hpx::future<std::int32_t> f2 =
           hpx::async(exec, hpx::util::bind(mult, _1), 42);
        HPX_TEST_EQ(f2.get(), 84);

        do_nothing_obj do_nothing_f;
        hpx::future<void> f3 = hpx::async(exec,
            hpx::util::bind(do_nothing_f, _1), 42);
        f3.get();
    }

    {
        decrement dec;

        hpx::future<std::int32_t> f1 =
            hpx::async(exec, &decrement::call, dec, 42);
        HPX_TEST_EQ(f1.get(), 41);

        do_nothing_member dnm;
        hpx::future<void> f2 = hpx::async(exec,
            &do_nothing_member::call, dnm, 42);
        f2.get();
    }

    {
        decrement dec;

        using hpx::util::placeholders::_1;

        hpx::future<std::int32_t> f1 =
            hpx::async(exec, hpx::util::bind(&decrement::call, dec, 42));
        HPX_TEST_EQ(f1.get(), 41);

        hpx::future<std::int32_t> f2 =
            hpx::async(exec, hpx::util::bind(&decrement::call, dec, _1), 42);
        HPX_TEST_EQ(f2.get(), 41);

        do_nothing_member dnm;
        hpx::future<void> f3 = hpx::async(exec,
            hpx::util::bind(&do_nothing_member::call, dnm, _1), 42);
        f3.get();
    }
}

int hpx_main()
{
    {
        hpx::parallel::sequential_executor exec;
        test_async_with_executor(exec);
    }

    {
        hpx::parallel::parallel_executor exec;
        test_async_with_executor(exec);
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

