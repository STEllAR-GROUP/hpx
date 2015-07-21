//  Taken from the Boost.Thread library
//
// Copyright (C) 2001-2003 William E. Kempf
// Copyright (C) 2007-2008 Anthony Williams
// Copyright (C) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/local/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/topology.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/chrono.hpp>

namespace
{
    hpx::lcos::local::spinlock multiple_wake_mutex;
    hpx::lcos::local::condition_variable multiple_wake_cond;
    unsigned multiple_wake_count=0;

    void wait_for_condvar_and_increase_count()
    {
        boost::unique_lock<hpx::lcos::local::spinlock> lk(multiple_wake_mutex);
        multiple_wake_cond.wait(lk);
        ++multiple_wake_count;
    }

    void join_all(std::vector<hpx::thread>& group)
    {
        for (std::size_t i = 0; i < group.size(); ++i)
            group[i].join();
    }

}

///////////////////////////////////////////////////////////////////////////////
struct wait_for_flag
{
    hpx::lcos::local::spinlock mutex;
    hpx::lcos::local::condition_variable cond_var;
    bool flag;
    unsigned woken;

    wait_for_flag():
        flag(false),woken(0)
    {}

    struct check_flag
    {
        bool const& flag;

        check_flag(bool const& flag_):
            flag(flag_)
        {}

        bool operator()() const
        {
            return flag;
        }
    private:
        void operator=(check_flag&);
    };


    void wait_without_predicate()
    {
        boost::unique_lock<hpx::lcos::local::spinlock> lock(mutex);
        while(!flag)
        {
            cond_var.wait(lock);
        }
        ++woken;
    }

    void wait_with_predicate()
    {
        boost::unique_lock<hpx::lcos::local::spinlock> lock(mutex);
        cond_var.wait(lock,check_flag(flag));
        if(flag)
        {
            ++woken;
        }
    }

    void wait_until_without_predicate()
    {
        boost::chrono::system_clock::time_point const timeout =
            boost::chrono::system_clock::now() + boost::chrono::milliseconds(5);

        boost::unique_lock<hpx::lcos::local::spinlock> lock(mutex);
        while(!flag)
        {
            if(cond_var.wait_until(lock,timeout) == hpx::lcos::local::cv_status::timeout)
            {
                return;
            }
        }
        ++woken;
    }

    void wait_until_with_predicate()
    {
        boost::chrono::system_clock::time_point const timeout =
            boost::chrono::system_clock::now() + boost::chrono::milliseconds(5);

        boost::unique_lock<hpx::lcos::local::spinlock> lock(mutex);
        if(cond_var.wait_until(lock,timeout,check_flag(flag)) && flag)
        {
            ++woken;
        }
    }
    void relative_wait_until_with_predicate()
    {
        boost::unique_lock<hpx::lcos::local::spinlock> lock(mutex);
        if(cond_var.wait_for(lock,boost::chrono::milliseconds(5),check_flag(flag)) && flag)
        {
            ++woken;
        }
    }
};

void test_condition_notify_one_wakes_from_wait()
{
    wait_for_flag data;

    hpx::thread thread(&wait_for_flag::wait_without_predicate, boost::ref(data));

    {
        boost::unique_lock<hpx::lcos::local::spinlock> lock(data.mutex);
        data.flag=true;
        data.cond_var.notify_one();
    }

    thread.join();
    HPX_TEST(data.woken);
}

void test_condition_notify_one_wakes_from_wait_with_predicate()
{
    wait_for_flag data;

    hpx::thread thread(&wait_for_flag::wait_with_predicate, boost::ref(data));

    {
        boost::unique_lock<hpx::lcos::local::spinlock> lock(data.mutex);
        data.flag=true;
        data.cond_var.notify_one();
    }

    thread.join();
    HPX_TEST(data.woken);
}

void test_condition_notify_one_wakes_from_wait_until()
{
    wait_for_flag data;

    hpx::thread thread(&wait_for_flag::wait_until_without_predicate, boost::ref(data));

    {
        boost::unique_lock<hpx::lcos::local::spinlock> lock(data.mutex);
        data.flag=true;
        data.cond_var.notify_one();
    }

    thread.join();
    HPX_TEST(data.woken);
}

void test_condition_notify_one_wakes_from_wait_until_with_predicate()
{
    wait_for_flag data;

    hpx::thread thread(&wait_for_flag::wait_until_with_predicate, boost::ref(data));

    {
        boost::unique_lock<hpx::lcos::local::spinlock> lock(data.mutex);
        data.flag=true;
        data.cond_var.notify_one();
    }

    thread.join();
    HPX_TEST(data.woken);
}

void test_condition_notify_one_wakes_from_relative_wait_until_with_predicate()
{
    wait_for_flag data;

    hpx::thread thread(&wait_for_flag::relative_wait_until_with_predicate, boost::ref(data));

    {
        boost::unique_lock<hpx::lcos::local::spinlock> lock(data.mutex);
        data.flag=true;
        data.cond_var.notify_one();
    }

    thread.join();
    HPX_TEST(data.woken);
}

void test_multiple_notify_one_calls_wakes_multiple_threads()
{
    multiple_wake_count=0;

    hpx::thread thread1(wait_for_condvar_and_increase_count);
    hpx::thread thread2(wait_for_condvar_and_increase_count);

    hpx::this_thread::sleep_for(boost::chrono::milliseconds(200));
    multiple_wake_cond.notify_one();

    hpx::thread thread3(wait_for_condvar_and_increase_count);

    hpx::this_thread::sleep_for(boost::chrono::milliseconds(200));
    multiple_wake_cond.notify_one();
    multiple_wake_cond.notify_one();
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(200));

    {
        boost::unique_lock<hpx::lcos::local::spinlock> lk(multiple_wake_mutex);
        HPX_TEST(multiple_wake_count==3);
    }

    thread1.join();
    thread2.join();
    thread3.join();
}

///////////////////////////////////////////////////////////////////////////////

void test_condition_notify_all_wakes_from_wait()
{
    wait_for_flag data;

    std::vector<hpx::thread> group;

    try
    {
        for(unsigned i=0;i<5;++i)
        {
            group.push_back(hpx::thread(&wait_for_flag::wait_without_predicate, boost::ref(data)));
        }

        {
            boost::unique_lock<hpx::lcos::local::spinlock> lock(data.mutex);
            data.flag=true;
            data.cond_var.notify_all();
        }

        join_all(group);
        HPX_TEST_EQ(data.woken,5u);
    }
    catch(...)
    {
        join_all(group);
        throw;
    }
}

void test_condition_notify_all_wakes_from_wait_with_predicate()
{
    wait_for_flag data;

    std::vector<hpx::thread> group;

    try
    {
        for(unsigned i=0;i<5;++i)
        {
            group.push_back(hpx::thread(&wait_for_flag::wait_with_predicate, boost::ref(data)));
        }

        {
            boost::unique_lock<hpx::lcos::local::spinlock> lock(data.mutex);
            data.flag=true;
            data.cond_var.notify_all();
        }

        join_all(group);
        HPX_TEST_EQ(data.woken,5u);
    }
    catch(...)
    {
        join_all(group);
        throw;
    }
}

void test_condition_notify_all_wakes_from_wait_until()
{
    wait_for_flag data;

    std::vector<hpx::thread> group;

    try
    {
        for(unsigned i=0;i<5;++i)
        {
            group.push_back(hpx::thread(&wait_for_flag::wait_until_without_predicate, boost::ref(data)));
        }

        {
            boost::unique_lock<hpx::lcos::local::spinlock> lock(data.mutex);
            data.flag=true;
            data.cond_var.notify_all();
        }

        join_all(group);
        HPX_TEST_EQ(data.woken,5u);
    }
    catch(...)
    {
        join_all(group);
        throw;
    }
}

void test_condition_notify_all_wakes_from_wait_until_with_predicate()
{
    wait_for_flag data;

    std::vector<hpx::thread> group;

    try
    {
        for(unsigned i=0;i<5;++i)
        {
            group.push_back(hpx::thread(&wait_for_flag::wait_until_with_predicate, boost::ref(data)));
        }

        {
            boost::unique_lock<hpx::lcos::local::spinlock> lock(data.mutex);
            data.flag=true;
            data.cond_var.notify_all();
        }

        join_all(group);
        HPX_TEST_EQ(data.woken,5u);
    }
    catch(...)
    {
        join_all(group);
        throw;
    }
}

void test_condition_notify_all_wakes_from_relative_wait_until_with_predicate()
{
    wait_for_flag data;

    std::vector<hpx::thread> group;

    try
    {
        for(unsigned i=0;i<5;++i)
        {
            group.push_back(hpx::thread(&wait_for_flag::relative_wait_until_with_predicate, boost::ref(data)));
        }

        {
            boost::unique_lock<hpx::lcos::local::spinlock> lock(data.mutex);
            data.flag=true;
            data.cond_var.notify_all();
        }

        join_all(group);
        HPX_TEST_EQ(data.woken,5u);
    }
    catch(...)
    {
        join_all(group);
        throw;
    }
}

void test_notify_all_following_notify_one_wakes_all_threads()
{
    multiple_wake_count=0;

    hpx::thread thread1(wait_for_condvar_and_increase_count);
    hpx::thread thread2(wait_for_condvar_and_increase_count);

    hpx::this_thread::sleep_for(boost::chrono::milliseconds(200));
    multiple_wake_cond.notify_one();

    hpx::thread thread3(wait_for_condvar_and_increase_count);

    hpx::this_thread::sleep_for(boost::chrono::milliseconds(200));
    multiple_wake_cond.notify_one();
    multiple_wake_cond.notify_all();
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(200));

    {
        boost::unique_lock<hpx::lcos::local::spinlock> lk(multiple_wake_mutex);
        HPX_TEST(multiple_wake_count==3);
    }

    thread1.join();
    thread2.join();
    thread3.join();
}

///////////////////////////////////////////////////////////////////////////////
struct condition_test_data
{
    condition_test_data() : notified(0), awoken(0) { }

    hpx::lcos::local::spinlock mutex;
    hpx::lcos::local::condition_variable condition;
    int notified;
    int awoken;
};

void condition_test_thread(condition_test_data* data)
{
    boost::unique_lock<hpx::lcos::local::spinlock> lock(data->mutex);
    HPX_TEST(lock ? true : false);
    while (!(data->notified > 0))
        data->condition.wait(lock);
    HPX_TEST(lock ? true : false);
    data->awoken++;
}

struct cond_predicate
{
    cond_predicate(int& var, int val) : _var(var), _val(val) { }

    bool operator()() { return _var == _val; }

    int& _var;
    int _val;
private:
    void operator=(cond_predicate&);

};

void condition_test_waits(condition_test_data* data)
{
    boost::unique_lock<hpx::lcos::local::spinlock> lock(data->mutex);
    HPX_TEST(lock ? true : false);

    // Test wait.
    while (data->notified != 1)
        data->condition.wait(lock);
    HPX_TEST(lock ? true : false);
    HPX_TEST_EQ(data->notified, 1);
    data->awoken++;
    data->condition.notify_one();

    // Test predicate wait.
    data->condition.wait(lock, cond_predicate(data->notified, 2));
    HPX_TEST(lock ? true : false);
    HPX_TEST_EQ(data->notified, 2);
    data->awoken++;
    data->condition.notify_one();

    // Test wait_until.
    boost::chrono::system_clock::time_point xt =
        boost::chrono::system_clock::now()
      + boost::chrono::milliseconds(10);
    while (data->notified != 3)
        data->condition.wait_until(lock, xt);
    HPX_TEST(lock ? true : false);
    HPX_TEST_EQ(data->notified, 3);
    data->awoken++;
    data->condition.notify_one();

    // Test predicate wait_until.
    xt = boost::chrono::system_clock::now()
      + boost::chrono::milliseconds(10);
    cond_predicate pred(data->notified, 4);
    HPX_TEST(data->condition.wait_until(lock, xt, pred));
    HPX_TEST(lock ? true : false);
    HPX_TEST(pred());
    HPX_TEST_EQ(data->notified, 4);
    data->awoken++;
    data->condition.notify_one();

    // Test predicate wait_for
    cond_predicate pred_rel(data->notified, 5);
    HPX_TEST(data->condition.wait_for(lock, boost::chrono::milliseconds(10), pred_rel));
    HPX_TEST(lock ? true : false);
    HPX_TEST(pred_rel());
    HPX_TEST_EQ(data->notified, 5);
    data->awoken++;
    data->condition.notify_one();
}

void test_condition_waits()
{
    typedef boost::unique_lock<hpx::lcos::local::spinlock> unique_lock;

    condition_test_data data;

    hpx::thread thread(&condition_test_waits, &data);

    {
        unique_lock lock(data.mutex);
        HPX_TEST(lock ? true : false);

        {
            hpx::util::unlock_guard<unique_lock> ul(lock);
            hpx::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }

        data.notified++;
        data.condition.notify_one();
        while (data.awoken != 1)
            data.condition.wait(lock);
        HPX_TEST(lock ? true : false);
        HPX_TEST_EQ(data.awoken, 1);

        {
            hpx::util::unlock_guard<unique_lock> ul(lock);
            hpx::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }

        data.notified++;
        data.condition.notify_one();
        while (data.awoken != 2)
            data.condition.wait(lock);
        HPX_TEST(lock ? true : false);
        HPX_TEST_EQ(data.awoken, 2);

        {
            hpx::util::unlock_guard<unique_lock> ul(lock);
            hpx::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }

        data.notified++;
        data.condition.notify_one();
        while (data.awoken != 3)
            data.condition.wait(lock);
        HPX_TEST(lock ? true : false);
        HPX_TEST_EQ(data.awoken, 3);

        {
            hpx::util::unlock_guard<unique_lock> ul(lock);
            hpx::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }

        data.notified++;
        data.condition.notify_one();
        while (data.awoken != 4)
            data.condition.wait(lock);
        HPX_TEST(lock ? true : false);
        HPX_TEST_EQ(data.awoken, 4);


        {
            hpx::util::unlock_guard<unique_lock> ul(lock);
            hpx::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }

        data.notified++;
        data.condition.notify_one();
        while (data.awoken != 5)
            data.condition.wait(lock);
        HPX_TEST(lock ? true : false);
        HPX_TEST_EQ(data.awoken, 5);
    }

    thread.join();
    HPX_TEST_EQ(data.awoken, 5);
}

///////////////////////////////////////////////////////////////////////////////

bool fake_predicate()
{
    return false;
}

boost::chrono::milliseconds const delay(1000);
boost::chrono::milliseconds const timeout_resolution(100);

void test_wait_until_times_out()
{
    hpx::lcos::local::condition_variable cond;
    hpx::lcos::local::spinlock m;

    boost::unique_lock<hpx::lcos::local::spinlock> lock(m);
    boost::chrono::system_clock::time_point const start =
        boost::chrono::system_clock::now();
    boost::chrono::system_clock::time_point const timeout = start + delay;

    while(cond.wait_until(lock, timeout) == hpx::lcos::local::cv_status::no_timeout) {}

    boost::chrono::system_clock::time_point const end =
        boost::chrono::system_clock::now();
    HPX_TEST_LTE(delay - timeout_resolution, end - start);
}

void test_wait_until_with_predicate_times_out()
{
    hpx::lcos::local::condition_variable cond;
    hpx::lcos::local::spinlock m;

    boost::unique_lock<hpx::lcos::local::spinlock> lock(m);
    boost::chrono::system_clock::time_point const start =
        boost::chrono::system_clock::now();
    boost::chrono::system_clock::time_point const timeout = start + delay;

    bool const res = cond.wait_until(lock, timeout, fake_predicate);

    boost::chrono::system_clock::time_point const end =
        boost::chrono::system_clock::now();
    HPX_TEST(!res);
    HPX_TEST_LTE(delay - timeout_resolution, end - start);
}

void test_relative_wait_until_with_predicate_times_out()
{
    hpx::lcos::local::condition_variable cond;
    hpx::lcos::local::spinlock m;

    boost::unique_lock<hpx::lcos::local::spinlock> lock(m);
    boost::chrono::system_clock::time_point const start =
        boost::chrono::system_clock::now();

    bool const res = cond.wait_for(lock, delay, fake_predicate);

    boost::chrono::system_clock::time_point const end =
        boost::chrono::system_clock::now();
    HPX_TEST(!res);
    HPX_TEST_LTE(delay - timeout_resolution, end - start);
}

void test_wait_until_relative_times_out()
{
    hpx::lcos::local::condition_variable cond;
    hpx::lcos::local::spinlock m;

    boost::unique_lock<hpx::lcos::local::spinlock> lock(m);
    boost::chrono::system_clock::time_point const start =
        boost::chrono::system_clock::now();

    while(cond.wait_for(lock, delay) == hpx::lcos::local::cv_status::no_timeout) {}

    boost::chrono::system_clock::time_point const end =
        boost::chrono::system_clock::now();
    HPX_TEST_LTE(delay - timeout_resolution, end - start);
}

///////////////////////////////////////////////////////////////////////////////
using boost::program_options::variables_map;
using boost::program_options::options_description;

int hpx_main(variables_map&)
{
    {
        test_condition_notify_one_wakes_from_wait();
        test_condition_notify_one_wakes_from_wait_with_predicate();
        test_condition_notify_one_wakes_from_wait_until();
        test_condition_notify_one_wakes_from_wait_until_with_predicate();
        test_condition_notify_one_wakes_from_relative_wait_until_with_predicate();
        test_multiple_notify_one_calls_wakes_multiple_threads();
    }
    {
        test_condition_notify_all_wakes_from_wait();
        test_condition_notify_all_wakes_from_wait_with_predicate();
        test_condition_notify_all_wakes_from_wait_until();
        test_condition_notify_all_wakes_from_wait_until_with_predicate();
        test_condition_notify_all_wakes_from_relative_wait_until_with_predicate();
        test_notify_all_following_notify_one_wakes_all_threads();
    }
    {
        test_condition_waits();
    }
    {
        test_wait_until_times_out();
        test_wait_until_with_predicate_times_out();
        test_relative_wait_until_with_predicate_times_out();
        test_wait_until_relative_times_out();
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}
