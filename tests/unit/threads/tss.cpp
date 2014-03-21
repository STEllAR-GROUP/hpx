// Copyright (C) 2001-2003 William E. Kempf
// Copyright (C) 2007 Anthony Williams
// Copyright (C) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

hpx::lcos::local::spinlock check_mutex;
hpx::lcos::local::spinlock tss_mutex;
int tss_instances = 0;
int tss_total = 0;

struct tss_value_t
{
    tss_value_t()
    {
        boost::unique_lock<hpx::lcos::local::spinlock> lock(tss_mutex);
        ++tss_instances;
        ++tss_total;
        value = 0;
    }
    ~tss_value_t()
    {
        boost::unique_lock<hpx::lcos::local::spinlock> lock(tss_mutex);
        --tss_instances;
    }
    int value;
};

hpx::threads::thread_specific_ptr<tss_value_t> tss_value;

void test_tss_thread()
{
    tss_value.reset(new tss_value_t());
    for (int i = 0; i < 1000; ++i)
    {
        int& n = tss_value->value;
        if (n != i)
        {
            boost::unique_lock<hpx::lcos::local::spinlock> lock(check_mutex);
            HPX_TEST_EQ(n, i);
        }
        ++n;
    }
}

void test_tss()
{
    tss_instances = 0;
    tss_total = 0;

    int const NUMTHREADS = 5;

    std::vector<hpx::unique_future<void> > threads;
    for (int i = 0; i < NUMTHREADS; ++i)
        threads.push_back(hpx::async(&test_tss_thread));
    hpx::wait_all(threads);

    HPX_TEST_EQ(tss_instances, 0);
    HPX_TEST_EQ(tss_total, 5);
}

///////////////////////////////////////////////////////////////////////////////
// bool tss_cleanup_called=false;
// 
// struct Dummy
// {};
// 
// void tss_custom_cleanup(Dummy* d)
// {
//     delete d;
//     tss_cleanup_called=true;
// }
// 
// boost::thread_specific_ptr<Dummy> tss_with_cleanup(tss_custom_cleanup);
// 
// void tss_thread_with_custom_cleanup()
// {
//     tss_with_cleanup.reset(new Dummy);
// }
// 
// void do_test_tss_with_custom_cleanup()
// {
//     boost::thread t(tss_thread_with_custom_cleanup);
//     try
//     {
//         t.join();
//     }
//     catch(...)
//     {
//         t.interrupt();
//         t.join();
//         throw;
//     }
// 
//     BOOST_CHECK(tss_cleanup_called);
// }
// 
// 
// void test_tss_with_custom_cleanup()
// {
//     timed_test(&do_test_tss_with_custom_cleanup, 2);
// }
// 
// Dummy* tss_object=new Dummy;
// 
// void tss_thread_with_custom_cleanup_and_release()
// {
//     tss_with_cleanup.reset(tss_object);
//     tss_with_cleanup.release();
// }
// 
// void do_test_tss_does_no_cleanup_after_release()
// {
//     tss_cleanup_called=false;
//     boost::thread t(tss_thread_with_custom_cleanup_and_release);
//     try
//     {
//         t.join();
//     }
//     catch(...)
//     {
//         t.interrupt();
//         t.join();
//         throw;
//     }
// 
//     BOOST_CHECK(!tss_cleanup_called);
//     if(!tss_cleanup_called)
//     {
//         delete tss_object;
//     }
// }
// 
// struct dummy_class_tracks_deletions
// {
//     static unsigned deletions;
// 
//     ~dummy_class_tracks_deletions()
//     {
//         ++deletions;
//     }
// 
// };
// 
// unsigned dummy_class_tracks_deletions::deletions=0;
// 
// boost::thread_specific_ptr<dummy_class_tracks_deletions> tss_with_null_cleanup(NULL);
// 
// void tss_thread_with_null_cleanup(dummy_class_tracks_deletions* delete_tracker)
// {
//     tss_with_null_cleanup.reset(delete_tracker);
// }
// 
// void do_test_tss_does_no_cleanup_with_null_cleanup_function()
// {
//     dummy_class_tracks_deletions* delete_tracker=new dummy_class_tracks_deletions;
//     boost::thread t(tss_thread_with_null_cleanup,delete_tracker);
//     try
//     {
//         t.join();
//     }
//     catch(...)
//     {
//         t.interrupt();
//         t.join();
//         throw;
//     }
// 
//     BOOST_CHECK(!dummy_class_tracks_deletions::deletions);
//     if(!dummy_class_tracks_deletions::deletions)
//     {
//         delete delete_tracker;
//     }
// }
// 
// void test_tss_does_no_cleanup_after_release()
// {
//     timed_test(&do_test_tss_does_no_cleanup_after_release, 2);
// }
// 
// void test_tss_does_no_cleanup_with_null_cleanup_function()
// {
//     timed_test(&do_test_tss_does_no_cleanup_with_null_cleanup_function, 2);
// }
// 
// void thread_with_local_tss_ptr()
// {
//     {
//         boost::thread_specific_ptr<Dummy> local_tss(tss_custom_cleanup);
// 
//         local_tss.reset(new Dummy);
//     }
//     BOOST_CHECK(tss_cleanup_called);
//     tss_cleanup_called=false;
// }
// 
// 
// void test_tss_does_not_call_cleanup_after_ptr_destroyed()
// {
//     boost::thread t(thread_with_local_tss_ptr);
//     t.join();
//     BOOST_CHECK(!tss_cleanup_called);
// }
// 
// void test_tss_cleanup_not_called_for_null_pointer()
// {
//     boost::thread_specific_ptr<Dummy> local_tss(tss_custom_cleanup);
//     local_tss.reset(new Dummy);
//     tss_cleanup_called=false;
//     local_tss.reset(0);
//     BOOST_CHECK(tss_cleanup_called);
//     tss_cleanup_called=false;
//     local_tss.reset(new Dummy);
//     BOOST_CHECK(!tss_cleanup_called);
// }
// 
// void test_tss_at_the_same_adress()
// {
//   for(int i=0; i<2; i++)
//   {
//     boost::thread_specific_ptr<Dummy> local_tss(tss_custom_cleanup);
//     local_tss.reset(new Dummy);
//     tss_cleanup_called=false;
//     BOOST_CHECK(tss_cleanup_called);
//     tss_cleanup_called=false;
//     BOOST_CHECK(!tss_cleanup_called);
//   }
// }


int main(int argc, char**argv)
{
    test_tss();
//     test->add(BOOST_TEST_CASE(test_tss));
//     test->add(BOOST_TEST_CASE(test_tss_with_custom_cleanup));
//     test->add(BOOST_TEST_CASE(test_tss_does_no_cleanup_after_release));
//     test->add(BOOST_TEST_CASE(test_tss_does_no_cleanup_with_null_cleanup_function));
//     test->add(BOOST_TEST_CASE(test_tss_does_not_call_cleanup_after_ptr_destroyed));
//     test->add(BOOST_TEST_CASE(test_tss_cleanup_not_called_for_null_pointer));

    return hpx::util::report_errors();
}
