// Copyright (c)       2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

struct test
{
    test() { ++count; }
    test(test const & t) { ++count; }
    test& operator=(test const & t) { ++count; return *this; }
    ~test() { --count; }

    static boost::atomic<int> count;
};

boost::atomic<int> test::count(0);

test call() { return test(); }
HPX_PLAIN_ACTION(call);

void test_leak()
{
    {
        hpx::shared_future<test> f;

        {
            hpx::lcos::promise<test> p;
            f = p.get_future();
            hpx::apply_c<call_action>(p.get_id(), hpx::find_here());
        }

        test t = f.get();
    }

    hpx::agas::garbage_collect();
    hpx::this_thread::yield();
    hpx::agas::garbage_collect();
    hpx::this_thread::yield();
    HPX_TEST_EQ(test::count, 0);
}

int hpx_main(int argc, char* argv[])
{
    {
        test_leak();

        hpx::id_type promise_id;
        hpx::future<int> f;
        {
            hpx::promise<int> p;
            f = p.get_future();
            {
                auto local_promise_id = p.get_id();
                hpx::cout << local_promise_id << hpx::endl;
            }

            hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));

            promise_id = p.get_id();
            hpx::cout << promise_id << hpx::endl;
        }

        hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));

        HPX_TEST(!f.is_ready());
        // This segfaults, because the promise is not alive any more.
        // It SHOULD get kept alive by AGAS though.
        hpx::set_lco_value(promise_id, 10, false);
        HPX_TEST(f.is_ready());
        HPX_TEST_EQ(f.get(), 10);
    }
    {
        hpx::id_type promise_id;
        {
            hpx::promise<int> p;
            p.get_future();
            {
                auto local_promise_id = p.get_id();
                hpx::cout << local_promise_id << hpx::endl;
            }

            hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));

            promise_id = p.get_id();
            hpx::cout << promise_id << hpx::endl;
        }

        hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));

        // This segfaults, because the promise is not alive any more.
        // It SHOULD get kept alive by AGAS though.
        hpx::set_lco_value(promise_id, 10, false);
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // initialize HPX, run hpx_main.
    hpx::start(argc, argv);

    // wait for hpx::finalize being called.
    return hpx::stop();
}
