//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_task_block.hpp>
#include <hpx/util/lightweight_test.hpp>

using hpx::parallel::define_task_block;
using hpx::parallel::task_block;
using hpx::parallel::par;
using hpx::parallel::task;

///////////////////////////////////////////////////////////////////////////////
void define_task_block_test()
{
    std::string s("test");

    bool parent_flag = false;
    bool task1_flag = false;
    bool task2_flag = false;
    bool task21_flag = false;
    bool task3_flag = false;

    define_task_block([&](task_block<>& trh)
    {
        parent_flag = true;

        trh.run([&]() {
            task1_flag = true;
            hpx::cout << "task1: " << s << hpx::endl;
        });

        trh.run([&]() {
            task2_flag = true;
            hpx::cout << "task2" << hpx::endl;

            define_task_block([&](task_block<>& trh) {
                trh.run([&]() {
                    task21_flag = true;
                    hpx::cout << "task2.1" << hpx::endl;
                });
            });
        });

        int i = 0, j = 10, k = 20;
        trh.run([=, &task3_flag]() {
            task3_flag = true;
            hpx::cout << "task3: " << i << " " << j << " " << k << hpx::endl;
        });

        hpx::cout << "parent" << hpx::endl;
    });

    HPX_TEST(parent_flag);
    HPX_TEST(task1_flag);
    HPX_TEST(task2_flag);
    HPX_TEST(task21_flag);
    HPX_TEST(task3_flag);
}

///////////////////////////////////////////////////////////////////////////////
void define_task_block_exceptions_test1()
{
    try {
        define_task_block([](task_block<>& trh) {
            trh.run([]() {
                hpx::cout << "task1" << hpx::endl;
                throw 1;
            });

            trh.run([]() {
                hpx::cout << "task2" << hpx::endl;
                throw 2;
            });

            hpx::cout << "parent" << hpx::endl;
            throw 100;
        });

        HPX_TEST(false);
    }
    catch (hpx::parallel::exception_list const& e) {
        HPX_TEST_EQ(e.size(), 3u);
    }
    catch(...) {
        HPX_TEST(false);
    }
}

void define_task_block_exceptions_test2()
{
    try {
        define_task_block([&](task_block<>& trh)
        {
            trh.run([&]()
            {
                HPX_TEST(!hpx::expect_exception());

                // Error: trh is not active
                trh.run([]()
                {
                    HPX_TEST(false);    // should not be called
                });

                HPX_TEST(false);

                HPX_TEST(hpx::expect_exception(false));
            });
        });

        HPX_TEST(false);
    }
    catch (hpx::exception const& e) {
        HPX_TEST_EQ(int(e.get_error()), int(hpx::task_block_not_active));
    }
    catch (...) {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    define_task_block_test();
    define_task_block_exceptions_test1();
    define_task_block_exceptions_test2();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}


