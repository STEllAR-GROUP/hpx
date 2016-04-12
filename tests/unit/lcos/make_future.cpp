//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
void test_make_future()
{
    // test make_future<T>(future<T>)
    {
        hpx::future<int> f1 = hpx::make_ready_future(42);
        hpx::future<int> f2 = hpx::make_future<int>(std::move(f1));
        HPX_TEST_EQ(42, f2.get());
    }

    // test make_future<T>(future<U>) where is_convertible<U, T>
    {
        hpx::future<int> f1 = hpx::make_ready_future(42);
        hpx::future<double> f2 = hpx::make_future<double>(std::move(f1));
        HPX_TEST_EQ(42.0, f2.get());
    }

    // test make_future<void>(future<U>)
    {
        hpx::future<int> f1 = hpx::make_ready_future(42);
        hpx::future<void> f2 = hpx::make_future<void>(std::move(f1));
    }

    // test make_future<void>(future<void>)
    {
        hpx::future<void> f1 = hpx::make_ready_future();
        hpx::future<void> f2 = hpx::make_future<void>(std::move(f1));
    }

    // test make_future<T>(future<U>) with given T conv(U)
    {
        hpx::future<int> f1 = hpx::make_ready_future(42);
        hpx::future<std::string> f2 =
            hpx::make_future<std::string>(
                std::move(f1),
                [](int value) -> std::string
                {
                    return std::to_string(value);
                });

        HPX_TEST_EQ(std::string("42"), f2.get());
    }
}

void test_make_shared_future()
{
    // test make_future<T>(shared_future<T>)
    {
        hpx::shared_future<int> f1 = hpx::make_ready_future(42);
        hpx::shared_future<int> f2 = hpx::make_future<int>(f1);
        HPX_TEST_EQ(42, f1.get());
        HPX_TEST_EQ(42, f2.get());
    }

    // test make_future<T>(shared_future<U>) where is_convertible<U, T>
    {
        hpx::shared_future<int> f1 = hpx::make_ready_future(42);
        hpx::shared_future<double> f2 = hpx::make_future<double>(f1);
        HPX_TEST_EQ(42, f1.get());
        HPX_TEST_EQ(42.0, f2.get());
    }

    // test make_future<void>(shared_future<U>)
    {
        hpx::shared_future<int> f1 = hpx::make_ready_future(42);
        hpx::shared_future<void> f2 = hpx::make_future<void>(f1);
        HPX_TEST_EQ(42, f1.get());
    }

    // test make_future<void>(shared_future<void>)
    {
        hpx::shared_future<void> f1 = hpx::make_ready_future();
        hpx::shared_future<void> f2 = hpx::make_future<void>(f1);
    }

    // test make_future<T>(shared_future<U>) with given T conv(U)
    {
        hpx::shared_future<int> f1 = hpx::make_ready_future(42);
        hpx::shared_future<std::string> f2 =
            hpx::make_future<std::string>(
                f1,
                [](int value) -> std::string
                {
                    return std::to_string(value);
                });

        HPX_TEST_EQ(42, f1.get());
        HPX_TEST_EQ(std::string("42"), f2.get());
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_make_future();
    test_make_shared_future();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
