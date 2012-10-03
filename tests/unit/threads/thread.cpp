// Copyright (C) 2012 Hartmut Kaiser
// Copyright (C) 2001-2003 William E. Kempf
// Copyright (C) 2008 Anthony Williams
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/lexical_cast.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

///////////////////////////////////////////////////////////////////////////////
inline void set_description(char const* test_name)
{
    hpx::threads::set_thread_description(hpx::threads::get_self_id(), test_name);
}

///////////////////////////////////////////////////////////////////////////////
inline int
time_cmp(boost::posix_time::ptime const& xt1, boost::posix_time::ptime const& xt2)
{
    if (xt1 == xt2)
        return 0;
    return xt1 > xt2 ? 1 : -1;
}

inline bool in_range(boost::posix_time::ptime const& xt, int secs = 1)
{
    boost::posix_time::ptime now =
        boost::date_time::second_clock<boost::posix_time::ptime>::universal_time();
    boost::posix_time::ptime mint(now + boost::posix_time::seconds(-secs));
    return time_cmp(xt, mint) >= 0 && time_cmp(xt, now) <= 0;
}

///////////////////////////////////////////////////////////////////////////////
template <typename F>
void timed_test(F func, int /*secs*/)
{
    hpx::thread thrd(func);
    thrd.join();

    // FIXME: implement execution monitor to verify in-time execution and to
    //        prevent deadlocks
}

///////////////////////////////////////////////////////////////////////////////
int test_value = 0;

void simple_thread()
{
    test_value = 999;
}

void comparison_thread(hpx::thread::id parent)
{
    hpx::thread::id const my_id = hpx::this_thread::get_id();
    HPX_TEST_NEQ(my_id, parent);

    hpx::thread::id const my_id2 = hpx::this_thread::get_id();
    HPX_TEST_EQ(my_id, my_id2);

    hpx::thread::id const no_thread_id = hpx::thread::id();
    HPX_TEST_NEQ(my_id, no_thread_id);
}

///////////////////////////////////////////////////////////////////////////////
void test_sleep()
{
    set_description("test_sleep");

    boost::posix_time::ptime now =
        boost::date_time::second_clock<boost::posix_time::ptime>::universal_time();
    hpx::this_thread::sleep_for(boost::posix_time::seconds(3));

    // Ensure it's in a range instead of checking actual equality due to time
    // lapse
    HPX_TEST(in_range(now, 4));
}

///////////////////////////////////////////////////////////////////////////////
void do_test_creation()
{
    test_value = 0;
    hpx::thread thrd(&simple_thread);
    thrd.join();
    HPX_TEST_EQ(test_value, 999);
}

void test_creation()
{
    set_description("test_creation");
    timed_test(&do_test_creation, 1);
}

///////////////////////////////////////////////////////////////////////////////
void do_test_id_comparison()
{
    hpx::thread::id const self = hpx::this_thread::get_id();
    hpx::thread thrd(HPX_STD_BIND(&comparison_thread, self));
    thrd.join();
}

void test_id_comparison()
{
    set_description("test_id_comparison");
    timed_test(&do_test_id_comparison, 1);
}

///////////////////////////////////////////////////////////////////////////////
void interruption_point_thread(hpx::lcos::local::barrier* b, 
    hpx::lcos::local::mutex* m, bool* failed)
{
    try {
        hpx::lcos::local::mutex::scoped_lock lk(*m);
        hpx::this_thread::interruption_point();
        *failed = true;
    }
    catch(...) {
        b->wait();
        throw;
    }
    b->wait();
}

void do_test_thread_interrupts_at_interruption_point()
{
    hpx::lcos::local::mutex m;
    hpx::lcos::local::barrier b(2);
    bool failed = false;
    hpx::lcos::local::mutex::scoped_lock lk(m);
    hpx::thread thrd(
        HPX_STD_BIND(&interruption_point_thread, &b, &m, &failed));
    thrd.interrupt();
    lk.unlock();

    b.wait();       // Make sure the test thread has been executed, as join is 
                    // a interruption point which might get triggered.

    thrd.join();
    HPX_TEST(!failed);
}

void test_thread_interrupts_at_interruption_point()
{
    set_description("test_thread_interrupts_at_interruption_point");
    timed_test(&do_test_thread_interrupts_at_interruption_point, 1);
}

///////////////////////////////////////////////////////////////////////////////
void disabled_interruption_point_thread(hpx::lcos::local::mutex* m, 
    hpx::lcos::local::barrier* b, bool* failed)
{
    hpx::this_thread::disable_interruption dc;
    try {
        hpx::lcos::local::mutex::scoped_lock lk(*m);
        hpx::this_thread::interruption_point();
        *failed = false;
    }
    catch(...) {
        b->wait();
        throw;
    }
    b->wait();
}

void do_test_thread_no_interrupt_if_interrupts_disabled_at_interruption_point()
{
    hpx::lcos::local::mutex m;
    hpx::lcos::local::barrier b(2);
    bool failed = true;
    hpx::lcos::local::mutex::scoped_lock lk(m);
    hpx::thread thrd(
        HPX_STD_BIND(&disabled_interruption_point_thread, &m, &b, &failed));
    thrd.interrupt();
    lk.unlock();

    b.wait();       // Make sure the test thread has been executed, as join is 
                    // a interruption point which might get triggered.

    thrd.join();
    HPX_TEST(!failed);
}

void test_thread_no_interrupt_if_interrupts_disabled_at_interruption_point()
{
    set_description("test_thread_no_interrupt_if_interrupts_disabled_at_interruption_point");
    timed_test(&do_test_thread_no_interrupt_if_interrupts_disabled_at_interruption_point, 1);
}

///////////////////////////////////////////////////////////////////////////////
struct non_copyable_functor:
    boost::noncopyable
{
    unsigned value;

    non_copyable_functor()
      : value(0)
    {}

    void operator()()
    {
        value = 999;
    }
};

void do_test_creation_through_reference_wrapper()
{
    non_copyable_functor f;

    hpx::thread thrd(boost::ref(f));
    thrd.join();
    HPX_TEST_EQ(f.value, 999u);
}

void test_creation_through_reference_wrapper()
{
    set_description("test_creation_through_reference_wrapper");
    timed_test(&do_test_creation_through_reference_wrapper, 1);
}

///////////////////////////////////////////////////////////////////////////////
// struct long_running_thread
// {
//     boost::condition_variable cond;
//     boost::mutex mut;
//     bool done;
//
//     long_running_thread()
//       : done(false)
//     {}
//
//     void operator()()
//     {
//         boost::mutex::scoped_lock lk(mut);
//         while(!done)
//         {
//             cond.wait(lk);
//         }
//     }
// };
//
// void do_test_timed_join()
// {
//     long_running_thread f;
//     hpx::thread thrd(boost::ref(f));
//     HPX_TEST(thrd.joinable());
//     boost::system_time xt=delay(3);
//     bool const joined=thrd.timed_join(xt);
//     HPX_TEST(in_range(boost::get_xtime(xt), 2));
//     HPX_TEST(!joined);
//     HPX_TEST(thrd.joinable());
//     {
//         boost::mutex::scoped_lock lk(f.mut);
//         f.done=true;
//         f.cond.notify_one();
//     }
//
//     xt=delay(3);
//     bool const joined2=thrd.timed_join(xt);
//     boost::system_time const now=boost::get_system_time();
//     HPX_TEST(xt>now);
//     HPX_TEST(joined2);
//     HPX_TEST(!thrd.joinable());
// }
//
// void test_timed_join()
// {
//     timed_test(&do_test_timed_join, 10);
// }

void simple_sync_thread(hpx::lcos::local::barrier& b1, hpx::lcos::local::barrier& b2)
{
    b1.wait();   // wait for both threads to be started
    // ... do nothing
    b2.wait();   // wait for the tests to be completed
}

void test_swap()
{
    set_description("test_swap");

    hpx::lcos::local::barrier b1(3);
    hpx::lcos::local::barrier b2(3);
    hpx::thread t1(hpx::util::bind(&simple_sync_thread, boost::ref(b1), boost::ref(b2)));
    hpx::thread t2(hpx::util::bind(&simple_sync_thread, boost::ref(b1), boost::ref(b2)));

    b1.wait();   // wait for both threads to be started

    hpx::thread::id id1 = t1.get_id();
    hpx::thread::id id2 = t2.get_id();

    t1.swap(t2);
    HPX_TEST(t1.get_id() == id2);
    HPX_TEST(t2.get_id() == id1);

    swap(t1, t2);
    HPX_TEST(t1.get_id() == id1);
    HPX_TEST(t2.get_id() == id2);

    b2.wait();   // wait for the tests to be completed

    t1.join();
    t2.join();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    {
        test_sleep();
        test_creation();
        test_id_comparison();
        test_thread_interrupts_at_interruption_point();
        test_thread_no_interrupt_if_interrupts_disabled_at_interruption_point();
        test_creation_through_reference_wrapper();
        test_swap();
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv);
}

