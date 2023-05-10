//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <chrono>
#include <utility>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
void test_interrupt_by_destructor()
{
    auto interval = std::chrono::milliseconds(200);
    bool was_interrupted = false;

    {
        hpx::jthread t([interval, &was_interrupted](hpx::stop_token stoken) {
            HPX_TEST(!stoken.stop_requested());
            try
            {
                // loop until interrupted (at most 40 times the interval)
                for (int i = 0; i < 40; ++i)
                {
                    if (stoken.stop_requested())
                    {
                        throw "interrupted";
                    }

                    hpx::this_thread::sleep_for(interval);
                }
                HPX_TEST(false);
            }
            catch (std::exception&)
            {
                // "interrupted" not derived from std::exception
                HPX_TEST(false);
            }
            catch (const char*)
            {
                HPX_TEST(stoken.stop_requested());
                was_interrupted = true;
            }
            catch (...)
            {
                HPX_TEST(false);
            }
        });

        HPX_TEST(!t.get_stop_source().stop_requested());

        // call destructor after 4 times the interval (should signal the interrupt)
        hpx::this_thread::sleep_for(4 * interval);
        HPX_TEST(!t.get_stop_source().stop_requested());
    }

    // key HPX_TESTion: signaled interrupt was processed
    HPX_TEST(was_interrupted);
}

///////////////////////////////////////////////////////////////////////////////
void test_interrupt_started_thread()
{
    auto interval = std::chrono::milliseconds(200);

    {
        bool interrupted = false;
        hpx::jthread t([interval, &interrupted](hpx::stop_token stoken) {
            try
            {
                // loop until interrupted (at most 40 times the interval)
                for (int i = 0; i < 40; ++i)
                {
                    if (stoken.stop_requested())
                    {
                        throw "interrupted";
                    }
                    hpx::this_thread::sleep_for(interval);
                }
                HPX_TEST(false);
            }
            catch (...)
            {
                interrupted = true;
            }
        });

        hpx::this_thread::sleep_for(4 * interval);
        t.request_stop();
        HPX_TEST(t.get_stop_source().stop_requested());
        t.join();
        HPX_TEST(interrupted);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_interrupt_started_thread_with_subthread()
{
    auto interval = std::chrono::milliseconds(200);

    {
        hpx::jthread t([interval](hpx::stop_token stoken) {
            hpx::jthread t2([interval, stoken] {
                while (!stoken.stop_requested())
                {
                    hpx::this_thread::sleep_for(interval);
                }
            });

            while (!stoken.stop_requested())
            {
                hpx::this_thread::sleep_for(interval);
            }
        });

        hpx::this_thread::sleep_for(4 * interval);
        t.request_stop();
        HPX_TEST(t.get_stop_source().stop_requested());
        t.join();
    }
}

////////////////////////////////////////////////////////////////////////////////
void test_basic_api_with_func()
{
    hpx::stop_source ssource;
    HPX_TEST(ssource.stop_possible());
    HPX_TEST(!ssource.stop_requested());

    {
        hpx::jthread t([]() {});
        ssource = t.get_stop_source();
        HPX_TEST(ssource.stop_possible());
        HPX_TEST(!ssource.stop_requested());
        hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    HPX_TEST(ssource.stop_possible());
    HPX_TEST(ssource.stop_requested());
}

////////////////////////////////////////////////////////////////////////////////
void test_exchange_token()
{
    auto interval = std::chrono::milliseconds(500);

    {
        std::atomic<hpx::stop_token*> pstoken(nullptr);
        hpx::jthread t([&pstoken](hpx::stop_token sstoken) {
            auto act_token = sstoken;
            int num_interrupts = 0;
            try
            {
                for (int i = 0; num_interrupts < 2 && i < 500; ++i)
                {
                    // if we get a new interrupt token from the caller, take it
                    if (pstoken.load() != nullptr)
                    {
                        act_token = *pstoken;
                        if (act_token.stop_requested())
                        {
                            ++num_interrupts;
                        }
                        pstoken.store(nullptr);
                    }
                    hpx::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
            catch (...)
            {
                HPX_TEST(false);
            }
        });

        hpx::this_thread::sleep_for(interval);
        t.request_stop();

        hpx::this_thread::sleep_for(interval);
        hpx::stop_token it;
        pstoken.store(&it);

        hpx::this_thread::sleep_for(interval);
        auto ssource2 = hpx::stop_source{};
        it = hpx::stop_token{ssource2.get_token()};
        pstoken.store(&it);

        hpx::this_thread::sleep_for(interval);
        ssource2.request_stop();

        hpx::this_thread::sleep_for(interval);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_concurrent_interrupt()
{
    int num_threads = 30;
    hpx::stop_source is;

    {
        hpx::jthread t1([it = is.get_token()](hpx::stop_token stoken) {
            try
            {
                bool stop_requested = false;
                for (int i = 0; !it.stop_requested(); ++i)
                {
                    // should never switch back once requested
                    if (stoken.stop_requested())
                    {
                        stop_requested = true;
                    }
                    else
                    {
                        HPX_TEST(!stop_requested);
                    }
                    hpx::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                HPX_TEST(stop_requested);
            }
            catch (...)
            {
                HPX_TEST(false);
            }
        });

        hpx::this_thread::sleep_for(std::chrono::milliseconds(500));

        // starts thread concurrently calling request_stop() for the same token
        std::vector<hpx::jthread> tv;
        int num_requested_stops = 0;
        for (int i = 0; i < num_threads; ++i)
        {
            hpx::this_thread::sleep_for(std::chrono::microseconds(100));
            hpx::jthread t([&t1, &num_requested_stops] {
                for (int i = 0; i < 13; ++i)
                {
                    // only first call to request_stop should return true
                    num_requested_stops += (t1.request_stop() ? 1 : 0);
                    HPX_TEST(!t1.request_stop());
                    hpx::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            });
            tv.push_back(std::move(t));
        }

        for (auto& t : tv)
        {
            t.join();
        }

        // only one request to request_stop() should have returned true
        HPX_TEST_EQ(num_requested_stops, 1);
        is.request_stop();
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_jthread_move()
{
    {
        bool interrupt_signalled = false;
        hpx::jthread t{[&interrupt_signalled](hpx::stop_token st) {
            while (!st.stop_requested())
            {
                hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (st.stop_requested())
            {
                interrupt_signalled = true;
            }
        }};

        hpx::jthread t2{std::move(t)};    // should compile

        // NOLINTNEXTLINE(bugprone-use-after-move)
        auto ssource = t.get_stop_source();
        HPX_TEST(!ssource.stop_possible());
        HPX_TEST(!ssource.stop_requested());

        ssource = t2.get_stop_source();
        HPX_TEST(ssource != hpx::stop_source{});
        HPX_TEST(ssource.stop_possible());
        HPX_TEST(!ssource.stop_requested());

        HPX_TEST(!interrupt_signalled);
        t.request_stop();
        HPX_TEST(!interrupt_signalled);
        t2.request_stop();
        t2.join();
        HPX_TEST(interrupt_signalled);
    }
}

///////////////////////////////////////////////////////////////////////////////
// void testEnabledIfForCopyConstructor_CompileTimeOnly()
// {
//     {
//         hpx::jthread t;
//         //hpx::jthread t2{t};  // should not compile
//     }
// }

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    std::set_terminate([]() { HPX_TEST(false); });

    test_interrupt_by_destructor();
    test_interrupt_started_thread();
    test_interrupt_started_thread_with_subthread();
    test_basic_api_with_func();
    test_exchange_token();
    test_concurrent_interrupt();
    test_jthread_move();
    //     testEnabledIfForCopyConstructor_CompileTimeOnly();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
