// Copyright (C) 2012 Hartmut Kaiser
//  (C) Copyright 2008-10 Anthony Williams
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <utility>
#include <memory>
#include <string>

#include <boost/move/move.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/assign/std/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
struct X
{
private:
    BOOST_MOVABLE_BUT_NOT_COPYABLE(X);

public:
    int i;

    X()
      : i(42)
    {}

    X(BOOST_RV_REF(X) other)
      : i(other.i)
    {
        other.i=0;
    }

    ~X() {}
};

int make_int()
{
    return 42;
}

int throw_runtime_error()
{
    throw std::runtime_error("42");
}

void set_promise_thread(hpx::lcos::local::promise<int>* p)
{
    p->set_value(42);
}

struct my_exception
{
};

void set_promise_exception_thread(hpx::lcos::local::promise<int>* p)
{
    p->set_exception(boost::copy_exception(my_exception()));
}

///////////////////////////////////////////////////////////////////////////////
void test_store_value_from_thread()
{
    hpx::lcos::local::promise<int> pi2;
    hpx::lcos::future<int> fi2 (pi2.get_future());
    hpx::thread(&set_promise_thread, &pi2);
    int j = fi2.get();
    HPX_TEST_EQ(j, 42);
    HPX_TEST(fi2.is_ready());
    HPX_TEST(fi2.has_value());
    HPX_TEST(!fi2.has_exception());
    HPX_TEST_EQ(fi2.get_state(), hpx::lcos::future_state::ready);
}

///////////////////////////////////////////////////////////////////////////////
void test_store_exception()
{
    hpx::lcos::local::promise<int> pi3;
    hpx::lcos::future<int> fi3 = pi3.get_future();
    hpx::thread(&set_promise_exception_thread, &pi3);
    try
    {
        fi3.get();
        HPX_TEST(false);
    }
    catch(my_exception)
    {
        HPX_TEST(true);
    }

    HPX_TEST(fi3.is_ready());
    HPX_TEST(!fi3.has_value());
    HPX_TEST(fi3.has_exception());
    HPX_TEST_EQ(fi3.get_state(), hpx::lcos::future_state::ready);
}

///////////////////////////////////////////////////////////////////////////////
void test_initial_state()
{
    hpx::lcos::future<int> fi;
    HPX_TEST(!fi.is_ready());
    HPX_TEST(!fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::uninitialized);
    int i;
    try
    {
        i = fi.get();
        HPX_TEST(false);
    }
    catch(boost::future_uninitialized)
    {
        HPX_TEST(true);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_waiting_future()
{
    hpx::lcos::local::promise<int> pi;
    hpx::lcos::future<int> fi;
    fi = pi.get_future();

    HPX_TEST(!fi.is_ready());
    HPX_TEST(!fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::deferred);
}

///////////////////////////////////////////////////////////////////////////////
void test_cannot_get_future_twice()
{
    hpx::lcos::local::promise<int> pi;
    pi.get_future();

    try {
        pi.get_future();
        HPX_TEST(false);
    }
    catch(boost::future_already_retrieved&) {
        HPX_TEST(true);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_set_value_updates_future_state()
{
    hpx::lcos::local::promise<int> pi;
    hpx::lcos::future<int> fi;
    fi = pi.get_future();

    pi.set_value(42);

    HPX_TEST(fi.is_ready());
    HPX_TEST(fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::ready);
}

///////////////////////////////////////////////////////////////////////////////
void test_set_value_can_be_retrieved()
{
    hpx::lcos::local::promise<int> pi;
    hpx::lcos::future<int> fi;
    fi = pi.get_future();

    pi.set_value(42);

    int i = fi.get();
    HPX_TEST_EQ(i, 42);
    HPX_TEST(fi.is_ready());
    HPX_TEST(fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::ready);
}

///////////////////////////////////////////////////////////////////////////////
void test_set_value_can_be_moved()
{
    hpx::lcos::local::promise<int> pi;
    hpx::lcos::future<int> fi;
    fi = pi.get_future();

    pi.set_value(42);

    int i=0;
    HPX_TEST(i=fi.get());
    HPX_TEST(i==42);
    HPX_TEST(fi.is_ready());
    HPX_TEST(fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST(fi.get_state()==hpx::lcos::future_state::ready);
}

///////////////////////////////////////////////////////////////////////////////
void test_future_from_packaged_task_is_waiting()
{
    hpx::lcos::local::packaged_task<int> pt(make_int);
    hpx::lcos::future<int> fi = pt.get_future();

    HPX_TEST(!fi.is_ready());
    HPX_TEST(!fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::deferred);
}

///////////////////////////////////////////////////////////////////////////////
void test_invoking_a_packaged_task_populates_future()
{
    hpx::lcos::local::packaged_task<int> pt(make_int);
    hpx::lcos::future<int> fi = pt.get_future();

    pt();

    HPX_TEST(fi.is_ready());
    HPX_TEST(fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::ready);

    int i = fi.get();
    HPX_TEST_EQ(i, 42);
}

///////////////////////////////////////////////////////////////////////////////
void test_invoking_a_packaged_task_twice_throws()
{
    hpx::lcos::local::packaged_task<int> pt(make_int);

    pt();
    try {
        pt();
        HPX_TEST(false);
    }
    catch(boost::task_already_started) {
        HPX_TEST(true);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_cannot_get_future_twice_from_task()
{
    hpx::lcos::local::packaged_task<int> pt(make_int);
    pt.get_future();
    try {
        pt.get_future();
        HPX_TEST(false);
    }
    catch(boost::future_already_retrieved) {
        HPX_TEST(true);
    }
}

void test_task_stores_exception_if_function_throws()
{
    hpx::lcos::local::packaged_task<int> pt(throw_runtime_error);
    hpx::lcos::future<int> fi = pt.get_future();

    pt();

    HPX_TEST(fi.is_ready());
    HPX_TEST(!fi.has_value());
    HPX_TEST(fi.has_exception());
    HPX_TEST(fi.get_state()==hpx::lcos::future_state::ready);
    try {
        fi.get();
        HPX_TEST(false);
    }
    catch (std::exception&) {
        HPX_TEST(true);
    }
    catch(...) {
        HPX_TEST(!"Unknown exception thrown");
    }

}

void test_void_promise()
{
    hpx::lcos::local::promise<void> p;
    hpx::lcos::future<void> f = p.get_future();

    p.set_value();
    HPX_TEST(f.is_ready());
    HPX_TEST(f.has_value());
    HPX_TEST(!f.has_exception());
    HPX_TEST_EQ(f.get_state(), hpx::lcos::future_state::ready);
    f.get();
}

// void test_reference_promise()
// {
//     hpx::lcos::local::promise<int&> p;
//     hpx::lcos::future<int&> f = p.get_future();
//     int i = 42;
//     p.set_value(i);
//     HPX_TEST(f.is_ready());
//     HPX_TEST(f.has_value());
//     HPX_TEST(!f.has_exception());
//     HPX_TEST_EQ(f.get_state(), hpx::lcos::future_state::ready);
//     HPX_TEST_EQ(&f.get(), &i);
// }

void do_nothing()
{
}

void test_task_returning_void()
{
    hpx::lcos::local::packaged_task<void> pt(do_nothing);
    hpx::lcos::future<void> fi = pt.get_future();

    pt();

    HPX_TEST(fi.is_ready());
    HPX_TEST(fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::ready);
}

// int global_ref_target = 0;
//
// int& return_ref()
// {
//     return global_ref_target;
// }
//
// void test_task_returning_reference()
// {
//     hpx::lcos::local::packaged_task<int&> pt(return_ref);
//     hpx::lcos::future<int&> fi = pt.get_future();
//
//     pt();
//
//     HPX_TEST(fi.is_ready());
//     HPX_TEST(fi.has_value());
//     HPX_TEST(!fi.has_exception());
//     HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::ready);
//     int& i = fi.get();
//     HPX_TEST_EQ(&i, &global_ref_target);
// }

void test_shared_future()
{
    hpx::lcos::local::packaged_task<int> pt(make_int);
    hpx::lcos::future<int> fi = pt.get_future();

    hpx::lcos::future<int> sf(boost::move(fi));
    HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::uninitialized);

    pt();

    HPX_TEST(sf.is_ready());
    HPX_TEST(sf.has_value());
    HPX_TEST(!sf.has_exception());
    HPX_TEST_EQ(sf.get_state(), hpx::lcos::future_state::ready);

    int i = sf.get();
    HPX_TEST_EQ(i, 42);
}

void test_copies_of_shared_future_become_ready_together()
{
    hpx::lcos::local::packaged_task<int> pt(make_int);
    hpx::lcos::future<int> fi=pt.get_future();

    hpx::lcos::future<int> sf1(boost::move(fi));
    hpx::lcos::future<int> sf2(sf1);
    hpx::lcos::future<int> sf3;

    sf3 = sf1;
    HPX_TEST_EQ(sf1.get_state(), hpx::lcos::future_state::deferred);
    HPX_TEST_EQ(sf2.get_state(), hpx::lcos::future_state::deferred);
    HPX_TEST_EQ(sf3.get_state(), hpx::lcos::future_state::deferred);

    pt();

    HPX_TEST(sf1.is_ready());
    HPX_TEST(sf1.has_value());
    HPX_TEST(!sf1.has_exception());
    HPX_TEST_EQ(sf1.get_state(), hpx::lcos::future_state::ready);
    int i = sf1.get();
    HPX_TEST_EQ(i, 42);

    i = 0;
    HPX_TEST(sf2.is_ready());
    HPX_TEST(sf2.has_value());
    HPX_TEST(!sf2.has_exception());
    HPX_TEST_EQ(sf2.get_state(), hpx::lcos::future_state::ready);
    i = sf2.get();
    HPX_TEST_EQ(i, 42);

    i = 0;
    HPX_TEST(sf3.is_ready());
    HPX_TEST(sf3.has_value());
    HPX_TEST(!sf3.has_exception());
    HPX_TEST_EQ(sf3.get_state(), hpx::lcos::future_state::ready);
    i = sf3.get();
    HPX_TEST_EQ(i, 42);
}

void test_shared_future_can_be_move_assigned_from_unique_future()
{
    hpx::lcos::local::packaged_task<int> pt(make_int);
    hpx::lcos::future<int> fi=pt.get_future();

    hpx::lcos::future<int> sf;
    sf = boost::move(fi);
    HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::uninitialized);

    HPX_TEST(!sf.is_ready());
    HPX_TEST(!sf.has_value());
    HPX_TEST(!sf.has_exception());
    HPX_TEST_EQ(sf.get_state(), hpx::lcos::future_state::deferred);
}

void test_shared_future_void()
{
    hpx::lcos::local::packaged_task<void> pt(do_nothing);
    hpx::lcos::future<void> fi=pt.get_future();

    hpx::lcos::future<void> sf(boost::move(fi));
    HPX_TEST_EQ(fi.get_state(), hpx::lcos::future_state::uninitialized);

    pt();

    HPX_TEST(sf.is_ready());
    HPX_TEST(sf.has_value());
    HPX_TEST(!sf.has_exception());
    HPX_TEST(sf.get_state()==hpx::lcos::future_state::ready);
    sf.get();
}

// void test_shared_future_ref()
// {
//     hpx::lcos::local::promise<int&> p;
//     hpx::lcos::future<int&> f(p.get_future());
//     int i = 42;
//     p.set_value(i);
//     HPX_TEST(f.is_ready());
//     HPX_TEST(f.has_value());
//     HPX_TEST(!f.has_exception());
//     HPX_TEST_EQ(f.get_state(), hpx::lcos::future_state::ready);
//     HPX_TEST_EQ(&f.get(), &i);
// }

void test_can_get_a_second_future_from_a_moved_promise()
{
    hpx::lcos::local::promise<int> pi;
    hpx::lcos::future<int> fi1 = pi.get_future();

    hpx::lcos::local::promise<int> pi2(boost::move(pi));
    hpx::lcos::future<int> fi2 = pi.get_future();

    pi2.set_value(3);
    HPX_TEST(fi1.is_ready());
    HPX_TEST(!fi2.is_ready());
    HPX_TEST_EQ(fi1.get(), 3);

    pi.set_value(42);
    HPX_TEST(fi2.is_ready());
    HPX_TEST_EQ(fi2.get(), 42);
}

void test_can_get_a_second_future_from_a_moved_void_promise()
{
    hpx::lcos::local::promise<void> pi;
    hpx::lcos::future<void> fi1 = pi.get_future();

    hpx::lcos::local::promise<void> pi2(boost::move(pi));
    hpx::lcos::future<void> fi2 = pi.get_future();

    pi2.set_value();
    HPX_TEST(fi1.is_ready());
    HPX_TEST(!fi2.is_ready());
    pi.set_value();
    HPX_TEST(fi2.is_ready());
}

// void test_unique_future_for_move_only_udt()
// {
//     hpx::lcos::local::promise<X> pt;
//     hpx::lcos::future<X> fi = pt.get_future();
//
//     pt.set_value(X());
//     X res(fi.get());
//     HPX_TEST_EQ(res.i, 42);
// }

void test_unique_future_for_string()
{
    hpx::lcos::local::promise<std::string> pt;
    hpx::lcos::future<std::string> fi1 = pt.get_future();

    pt.set_value(std::string("hello"));
    std::string res(fi1.get());
    HPX_TEST_EQ(res, "hello");

    hpx::lcos::local::promise<std::string> pt2;
    fi1 = pt2.get_future();

    std::string const s = "goodbye";

    pt2.set_value(s);
    res = fi1.get();
    HPX_TEST_EQ(res, "goodbye");

    hpx::lcos::local::promise<std::string> pt3;
    fi1 = pt3.get_future();

    std::string s2 = "foo";

    pt3.set_value(s2);
    res = fi1.get();
    HPX_TEST_EQ(res, "foo");
}

// hpx::lcos::local::mutex callback_mutex;
// unsigned callback_called = 0;
//
// void wait_callback(hpx::lcos::local::promise<int>& pi)
// {
//     boost::lock_guard<hpx::lcos::local::mutex> lk(callback_mutex);
//     ++callback_called;
//     try {
//         pi.set_value(42);
//     }
//     catch(...) {
//     }
// }
//
// void do_nothing_callback(hpx::lcos::local::promise<int>& /*pi*/)
// {
//     boost::lock_guard<hpx::lcos::local::mutex> lk(callback_mutex);
//     ++callback_called;
// }

// void test_wait_callback()
// {
//     callback_called = 0;
//     hpx::lcos::local::promise<int> pi;
//     hpx::lcos::future<int> fi = pi.get_future();
//     pi.set_wait_callback(wait_callback);
//     fi.wait();
//     HPX_TEST(callback_called);
//     HPX_TEST_EQ(fi.get(), 42);
//     fi.wait();
//     fi.wait();
//     HPX_TEST_EQ(callback_called, 1);
// }

// void test_wait_callback_with_timed_wait()
// {
//     callback_called=0;
//     hpx::lcos::local::promise<int> pi;
//     hpx::lcos::future<int> fi = pi.get_future();
//     pi.set_wait_callback(do_nothing_callback);
//     bool success = fi.timed_wait(boost::posix_time::milliseconds(10));
//     HPX_TEST(callback_called);
//     HPX_TEST(!success);
//     success = fi.timed_wait(boost::posix_time::milliseconds(10));
//     HPX_TEST(!success);
//     success = fi.timed_wait(boost::posix_time::milliseconds(10));
//     HPX_TEST(!success);
//     HPX_TEST_EQ(callback_called, 3);
//     pi.set_value(42);
//     success = fi.timed_wait(boost::posix_time::milliseconds(10));
//     HPX_TEST(success);
//     HPX_TEST_EQ(callback_called, 3);
//     HPX_TEST_EQ(fi.get(), 42);
//     HPX_TEST_EQ(callback_called, 3);
// }

// void wait_callback_for_task(hpx::lcos::local::packaged_task<int>& pt)
// {
//     boost::lock_guard<hpx::lcos::local::mutex> lk(callback_mutex);
//     ++callback_called;
//     try {
//         pt();
//     }
//     catch(...) {
//     }
// }

// void test_wait_callback_for_packaged_task()
// {
//     callback_called=0;
//     hpx::lcos::local::packaged_task<int> pt(make_int);
//     hpx::lcos::future<int> fi = pt.get_future();
//     pt.set_wait_callback(wait_callback_for_task);
//     fi.wait();
//     HPX_TEST(callback_called);
//     HPX_TEST_EQ(fi.get(), 42);
//     fi.wait();
//     fi.wait();
//     HPX_TEST_EQ(callback_called, 1);
// }

void test_packaged_task_can_be_moved()
{
    hpx::lcos::local::packaged_task<int> pt(make_int);
    hpx::lcos::future<int> fi = pt.get_future();
    HPX_TEST(!fi.is_ready());

    hpx::lcos::local::packaged_task<int> pt2(boost::move(pt));
    HPX_TEST(!fi.is_ready());

    try {
        pt();
        HPX_TEST(!"Can invoke moved task!");
    }
    catch(boost::task_moved&) {
    }

    HPX_TEST(!fi.is_ready());

    pt2();

    HPX_TEST(fi.is_ready());
}

void test_destroying_a_promise_stores_broken_promise()
{
    hpx::lcos::future<int> f;

    {
        hpx::lcos::local::promise<int> p;
        f = p.get_future();
    }

    HPX_TEST(f.is_ready());
    HPX_TEST(f.has_exception());
    try {
        f.get();
    }
    catch(boost::broken_promise&) {
    }
}

void test_destroying_a_packaged_task_stores_broken_promise()
{
    hpx::lcos::future<int> f;

    {
        hpx::lcos::local::packaged_task<int> p(make_int);
        f = p.get_future();
    }

    HPX_TEST(f.is_ready());
    HPX_TEST(f.has_exception());
    try {
        f.get();
    }
    catch(boost::broken_promise&) {
    }
}

int make_int_slowly()
{
    hpx::this_thread::sleep_for(boost::posix_time::seconds(1));
    return 42;
}

// void test_wait_for_either_of_two_futures_1()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//
//     hpx::thread(boost::move(pt));
//
//     hpx::lcos::future<std::pair<int, hpx::lcos::future<int> > r =
//        hpx::wait_any(f1, f2);
//
//     HPX_TEST(future, 0);
//     HPX_TEST(f1.is_ready());
//     HPX_TEST(!f2.is_ready());
//     HPX_TEST(f1.get(), 42);
// }

// void test_wait_for_either_of_two_futures_2()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//
//     hpx::thread(boost::move(pt2));
//
//     unsigned const future = boost::wait_for_any(f1,f2);
//
//     HPX_TEST_EQ(future, 1);
//     HPX_TEST(!f1.is_ready());
//     HPX_TEST(f2.is_ready());
//     HPX_TEST_EQ(f2.get(), 42);
// }

// void test_wait_for_either_of_three_futures_1()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//
//     hpx::thread(boost::move(pt));
//
//     unsigned const future = boost::wait_for_any(f1,f2,f3);
//
//     HPX_TEST_EQ(future, 0);
//     HPX_TEST(f1.is_ready());
//     HPX_TEST(!f2.is_ready());
//     HPX_TEST(!f3.is_ready());
//     HPX_TEST_EQ(f1.get(), 42);
// }

// void test_wait_for_either_of_three_futures_2()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//
//     hpx::thread(boost::move(pt2));
//
//     unsigned const future = boost::wait_for_any(f1,f2,f3);
//
//     HPX_TEST_EQ(future, 1);
//     HPX_TEST(!f1.is_ready());
//     HPX_TEST(f2.is_ready());
//     HPX_TEST(!f3.is_ready());
//     HPX_TEST_EQ(f2.get(), 42);
// }

// void test_wait_for_either_of_three_futures_3()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//
//     hpx::thread(boost::move(pt3));
//
//     unsigned const future = boost::wait_for_any(f1,f2,f3);
//
//     HPX_TEST_EQ(future, 2);
//     HPX_TEST(!f1.is_ready());
//     HPX_TEST(!f2.is_ready());
//     HPX_TEST(f3.is_ready());
//     HPX_TEST_EQ(f3.get(), 42);
// }

// void test_wait_for_either_of_four_futures_1()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//     hpx::lcos::local::packaged_task<int> pt4(make_int_slowly);
//     hpx::lcos::future<int> f4(pt4.get_future());
//
//     hpx::thread(boost::move(pt));
//
//     unsigned const future = boost::wait_for_any(f1,f2,f3,f4);
//
//     HPX_TEST_EQ(future, 0);
//     HPX_TEST(f1.is_ready());
//     HPX_TEST(!f2.is_ready());
//     HPX_TEST(!f3.is_ready());
//     HPX_TEST(!f4.is_ready());
//     HPX_TEST_EQ(f1.get(), 42);
// }

// void test_wait_for_either_of_four_futures_2()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//     hpx::lcos::local::packaged_task<int> pt4(make_int_slowly);
//     hpx::lcos::future<int> f4(pt4.get_future());
//
//     hpx::thread(boost::move(pt2));
//
//     unsigned const future = boost::wait_for_any(f1,f2,f3,f4);
//
//     HPX_TEST_EQ(future, 1);
//     HPX_TEST(!f1.is_ready());
//     HPX_TEST(f2.is_ready());
//     HPX_TEST(!f3.is_ready());
//     HPX_TEST(!f4.is_ready());
//     HPX_TEST_EQ(f2.get(), 42);
// }

// void test_wait_for_either_of_four_futures_3()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//     hpx::lcos::local::packaged_task<int> pt4(make_int_slowly);
//     hpx::lcos::future<int> f4(pt4.get_future());
//
//     hpx::thread(boost::move(pt3));
//
//     unsigned const future = boost::wait_for_any(f1, f2, f3, f4);
//
//     HPX_TEST_EQ(future, 2);
//     HPX_TEST(!f1.is_ready());
//     HPX_TEST(!f2.is_ready());
//     HPX_TEST(f3.is_ready());
//     HPX_TEST(!f4.is_ready());
//     HPX_TEST_EQ(f3.get(), 42);
// }

// void test_wait_for_either_of_four_futures_4()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//     hpx::lcos::local::packaged_task<int> pt4(make_int_slowly);
//     hpx::lcos::future<int> f4(pt4.get_future());
//
//     hpx::thread(boost::move(pt4));
//
//     unsigned const future=boost::wait_for_any(f1,f2,f3,f4);
//
//     HPX_TEST_EQ(future, 3);
//     HPX_TEST(!f1.is_ready());
//     HPX_TEST(!f2.is_ready());
//     HPX_TEST(!f3.is_ready());
//     HPX_TEST(f4.is_ready());
//     HPX_TEST_EQ(f4.get(), 42);
// }

// void test_wait_for_either_of_five_futures_1()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//     hpx::lcos::local::packaged_task<int> pt4(make_int_slowly);
//     hpx::lcos::future<int> f4(pt4.get_future());
//     hpx::lcos::local::packaged_task<int> pt5(make_int_slowly);
//     hpx::lcos::future<int> f5(pt5.get_future());
//
//     hpx::thread(boost::move(pt));
//
//     unsigned const future=boost::wait_for_any(f1,f2,f3,f4,f5);
//
//     HPX_TEST_EQ(future, 0);
//     HPX_TEST(f1.is_ready());
//     HPX_TEST(!f2.is_ready());
//     HPX_TEST(!f3.is_ready());
//     HPX_TEST(!f4.is_ready());
//     HPX_TEST(!f5.is_ready());
//     HPX_TEST_EQ(f1.get(), 42);
// }

// void test_wait_for_either_of_five_futures_2()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//     hpx::lcos::local::packaged_task<int> pt4(make_int_slowly);
//     hpx::lcos::future<int> f4(pt4.get_future());
//     hpx::lcos::local::packaged_task<int> pt5(make_int_slowly);
//     hpx::lcos::future<int> f5(pt5.get_future());
//
//     hpx::thread(boost::move(pt2));
//
//     unsigned const future = boost::wait_for_any(f1,f2,f3,f4,f5);
//
//     HPX_TEST_EQ(future, 1);
//     HPX_TEST(!f1.is_ready());
//     HPX_TEST(f2.is_ready());
//     HPX_TEST(!f3.is_ready());
//     HPX_TEST(!f4.is_ready());
//     HPX_TEST(!f5.is_ready());
//     HPX_TEST_EQ(f2.get(), 42);
// }

// void test_wait_for_either_of_five_futures_3()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//     hpx::lcos::local::packaged_task<int> pt4(make_int_slowly);
//     hpx::lcos::future<int> f4(pt4.get_future());
//     hpx::lcos::local::packaged_task<int> pt5(make_int_slowly);
//     hpx::lcos::future<int> f5(pt5.get_future());
//
//     hpx::thread(boost::move(pt3));
//
//     unsigned const future = boost::wait_for_any(f1,f2,f3,f4,f5);
//
//     HPX_TEST_EQ(future, 2);
//     HPX_TEST(!f1.is_ready());
//     HPX_TEST(!f2.is_ready());
//     HPX_TEST(f3.is_ready());
//     HPX_TEST(!f4.is_ready());
//     HPX_TEST(!f5.is_ready());
//     HPX_TEST_EQ(f3.get(), 42);
// }

// void test_wait_for_either_of_five_futures_4()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//     hpx::lcos::local::packaged_task<int> pt4(make_int_slowly);
//     hpx::lcos::future<int> f4(pt4.get_future());
//     hpx::lcos::local::packaged_task<int> pt5(make_int_slowly);
//     hpx::lcos::future<int> f5(pt5.get_future());
//
//     hpx::thread(boost::move(pt4));
//
//     unsigned const future = boost::wait_for_any(f1,f2,f3,f4,f5);
//
//     HPX_TEST_EQ(future, 3);
//     HPX_TEST(!f1.is_ready());
//     HPX_TEST(!f2.is_ready());
//     HPX_TEST(!f3.is_ready());
//     HPX_TEST(f4.is_ready());
//     HPX_TEST(!f5.is_ready());
//     HPX_TEST_EQ(f4.get(), 42);
// }

// void test_wait_for_either_of_five_futures_5()
// {
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> f1(pt.get_future());
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> f2(pt2.get_future());
//     hpx::lcos::local::packaged_task<int> pt3(make_int_slowly);
//     hpx::lcos::future<int> f3(pt3.get_future());
//     hpx::lcos::local::packaged_task<int> pt4(make_int_slowly);
//     hpx::lcos::future<int> f4(pt4.get_future());
//     hpx::lcos::local::packaged_task<int> pt5(make_int_slowly);
//     hpx::lcos::future<int> f5(pt5.get_future());
//
//     hpx::thread(boost::move(pt5));
//
//     unsigned const future = boost::wait_for_any(f1,f2,f3,f4,f5);
//
//     HPX_TEST_EQ(future, 4);
//     HPX_TEST(!f1.is_ready());
//     HPX_TEST(!f2.is_ready());
//     HPX_TEST(!f3.is_ready());
//     HPX_TEST(!f4.is_ready());
//     HPX_TEST(f5.is_ready());
//     HPX_TEST_EQ(f5.get(), 42);
// }

///////////////////////////////////////////////////////////////////////////////
// void test_wait_for_either_invokes_callbacks()
// {
//     callback_called = 0;
//     hpx::lcos::local::packaged_task<int> pt(make_int_slowly);
//     hpx::lcos::future<int> fi = pt.get_future();
//     hpx::lcos::local::packaged_task<int> pt2(make_int_slowly);
//     hpx::lcos::future<int> fi2 = pt2.get_future();
//     pt.set_wait_callback(wait_callback_for_task);
//
//     hpx::thread(boost::move(pt));
//
//     boost::wait_for_any(fi, fi2);
//     HPX_TEST_EQ(callback_called, 1);
//     HPX_TEST_EQ(fi.get(), 42);
// }

// void test_wait_for_any_from_range()
// {
//     unsigned const count = 10;
//     for(unsigned i = 0; i < count; ++i)
//     {
//         hpx::lcos::local::packaged_task<int> tasks[count];
//         hpx::lcos::future<int> futures[count];
//         for(unsigned j = 0; j < count; ++j)
//         {
//             tasks[j] = boost::move(hpx::lcos::local::packaged_task<int>(make_int_slowly));
//             futures[j] = tasks[j].get_future();
//         }
//         hpx::thread(boost::move(tasks[i]));
//
//         HPX_TEST_EQ(boost::wait_for_any(futures,futures), futures);
//
//         hpx::lcos::future<int>* const future = boost::wait_for_any(futures, futures+count);
//
//         HPX_TEST(future == (futures + i));
//         for(unsigned j = 0; j < count; ++j)
//         {
//             if (j != i)
//             {
//                 HPX_TEST(!futures[j].is_ready());
//             }
//             else
//             {
//                 HPX_TEST(futures[j].is_ready());
//             }
//         }
//         HPX_TEST_EQ(futures[i].get(), 42);
//     }
// }

// void test_wait_for_all_from_range()
// {
//     unsigned const count = 10;
//     hpx::lcos::future<int> futures[count];
//     for(unsigned j = 0; j < count; ++j)
//     {
//         hpx::lcos::local::packaged_task<int> task(make_int_slowly);
//         futures[j] = task.get_future();
//         hpx::thread(boost::move(task));
//     }
//
//     boost::wait_for_all(futures,futures+count);
//
//     for(unsigned j = 0; j < count; ++j)
//     {
//         HPX_TEST(futures[j].is_ready());
//     }
// }

// void test_wait_for_all_two_futures()
// {
//     unsigned const count = 2;
//     hpx::lcos::future<int> futures[count];
//     for(unsigned j = 0; j < count; ++j)
//     {
//         hpx::lcos::local::packaged_task<int> task(make_int_slowly);
//         futures[j]=task.get_future();
//         hpx::thread(boost::move(task));
//     }
//
//     boost::wait_for_all(futures[0],futures[1]);
//
//     for(unsigned j = 0; j < count; ++j)
//     {
//         HPX_TEST(futures[j].is_ready());
//     }
// }

// void test_wait_for_all_three_futures()
// {
//     unsigned const count=3;
//     hpx::lcos::future<int> futures[count];
//     for(unsigned j = 0; j < count; ++j)
//     {
//         hpx::lcos::local::packaged_task<int> task(make_int_slowly);
//         futures[j]=task.get_future();
//         hpx::thread(boost::move(task));
//     }
//
//     boost::wait_for_all(futures[0],futures[1],futures[2]);
//
//     for(unsigned j = 0; j < count; ++j)
//     {
//         HPX_TEST(futures[j].is_ready());
//     }
// }

// void test_wait_for_all_four_futures()
// {
//     unsigned const count=4;
//     hpx::lcos::future<int> futures[count];
//     for(unsigned j = 0; j < count; ++j)
//     {
//         hpx::lcos::local::packaged_task<int> task(make_int_slowly);
//         futures[j]=task.get_future();
//         hpx::thread(boost::move(task));
//     }
//
//     boost::wait_for_all(futures[0],futures[1],futures[2],futures[3]);
//
//     for(unsigned j = 0; j < count; ++j)
//     {
//         HPX_TEST(futures[j].is_ready());
//     }
// }

// void test_wait_for_all_five_futures()
// {
//     unsigned const count=5;
//     hpx::lcos::future<int> futures[count];
//     for(unsigned j = 0; j < count; ++j)
//     {
//         hpx::lcos::local::packaged_task<int> task(make_int_slowly);
//         futures[j]=task.get_future();
//         hpx::thread(boost::move(task));
//     }
//
//     boost::wait_for_all(futures[0],futures[1],futures[2],futures[3],futures[4]);
//
//     for(unsigned j = 0; j < count; ++j)
//     {
//         HPX_TEST(futures[j].is_ready());
//     }
// }

///////////////////////////////////////////////////////////////////////////////
using boost::program_options::variables_map;
using boost::program_options::options_description;

int hpx_main(variables_map&)
{
    {
        test_store_value_from_thread();
        test_store_exception();
        test_initial_state();
        test_waiting_future();
        test_cannot_get_future_twice();
        test_set_value_updates_future_state();
        test_set_value_can_be_retrieved();
        test_set_value_can_be_moved();
        test_future_from_packaged_task_is_waiting();
        test_invoking_a_packaged_task_populates_future();
        test_invoking_a_packaged_task_twice_throws();
        test_cannot_get_future_twice_from_task();
        test_task_stores_exception_if_function_throws();
        test_void_promise();
//         test_reference_promise();
        test_task_returning_void();
//         test_task_returning_reference();
        test_shared_future();
        test_copies_of_shared_future_become_ready_together();
        test_shared_future_can_be_move_assigned_from_unique_future();
        test_shared_future_void();
//         test_shared_future_ref();
        test_can_get_a_second_future_from_a_moved_promise();
        test_can_get_a_second_future_from_a_moved_void_promise();
//         test_unique_future_for_move_only_udt();
        test_unique_future_for_string();
//         test_wait_callback();
//         test_wait_callback_with_timed_wait();
//         test_wait_callback_for_packaged_task();
        test_packaged_task_can_be_moved();
        test_destroying_a_promise_stores_broken_promise();
        test_destroying_a_packaged_task_stores_broken_promise();
//         test_wait_for_either_of_two_futures_1();
//         test_wait_for_either_of_two_futures_2();
//         test_wait_for_either_of_three_futures_1();
//         test_wait_for_either_of_three_futures_2();
//         test_wait_for_either_of_three_futures_3();
//         test_wait_for_either_of_four_futures_1();
//         test_wait_for_either_of_four_futures_2();
//         test_wait_for_either_of_four_futures_3();
//         test_wait_for_either_of_four_futures_4();
//         test_wait_for_either_of_five_futures_1();
//         test_wait_for_either_of_five_futures_2();
//         test_wait_for_either_of_five_futures_3();
//         test_wait_for_either_of_five_futures_4();
//         test_wait_for_either_of_five_futures_5();
//         test_wait_for_either_invokes_callbacks();
//         test_wait_for_any_from_range();
//         test_wait_for_all_from_range();
//         test_wait_for_all_two_futures();
//         test_wait_for_all_three_futures();
//         test_wait_for_all_four_futures();
//         test_wait_for_all_five_futures();
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // we force this test to use several (4) threads
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<int>(hpx::thread::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}

