// Copyright (C) 2012 Hartmut Kaiser
//  (C) Copyright 2008-10 Anthony Williams
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define HPX_WAIT_ARGUMENT_LIMIT 5

#include <hpx/hpx_init.hpp>
#include <hpx/include/thread.hpp>
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
    hpx::thread t(&set_promise_thread, &pi2);
    int j = fi2.get();
    HPX_TEST_EQ(j, 42);
    HPX_TEST(fi2.is_ready());
    HPX_TEST(fi2.has_value());
    HPX_TEST(!fi2.has_exception());
    HPX_TEST(fi2.get_state() == hpx::lcos::future_status::ready);
    t.join();
}

///////////////////////////////////////////////////////////////////////////////
void test_store_exception()
{
    hpx::lcos::local::promise<int> pi3;
    hpx::lcos::future<int> fi3 = pi3.get_future();
    hpx::thread t(&set_promise_exception_thread, &pi3);
    try {
        fi3.get();
        HPX_TEST(false);
    }
    catch (my_exception) {
        HPX_TEST(true);
    }

    HPX_TEST(fi3.is_ready());
    HPX_TEST(!fi3.has_value());
    HPX_TEST(fi3.has_exception());
    HPX_TEST(fi3.get_state() ==  hpx::lcos::future_status::ready);
    t.join();
}

///////////////////////////////////////////////////////////////////////////////
void test_initial_state()
{
    hpx::lcos::future<int> fi;
    HPX_TEST(!fi.is_ready());
    HPX_TEST(!fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::uninitialized);
    int i;
    try {
        i = fi.get();
        HPX_TEST(false);
    }
    catch (hpx::exception const& e) {
        HPX_TEST(e.get_error() == hpx::future_uninitialized);
    }
    catch (...) {
        HPX_TEST(false);
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
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::deferred);
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
    catch (hpx::exception const& e) {
        HPX_TEST(e.get_error() == hpx::future_already_retrieved);
    }
    catch (...) {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_set_value_updates_future_status()
{
    hpx::lcos::local::promise<int> pi;
    hpx::lcos::future<int> fi;
    fi = pi.get_future();

    pi.set_value(42);

    HPX_TEST(fi.is_ready());
    HPX_TEST(fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::ready);
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
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::ready);
}

///////////////////////////////////////////////////////////////////////////////
void test_set_value_can_be_moved()
{
    hpx::lcos::local::promise<int> pi;
    hpx::lcos::future<int> fi;
    fi = pi.get_future();

    pi.set_value(42);

    int i=0;
    HPX_TEST(i = fi.get());
    HPX_TEST_EQ(i, 42);
    HPX_TEST(fi.is_ready());
    HPX_TEST(fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::ready);
}

///////////////////////////////////////////////////////////////////////////////
void test_future_from_packaged_task_is_waiting()
{
    hpx::lcos::local::packaged_task<int()> pt(make_int);
    hpx::lcos::future<int> fi = pt.get_future();

    HPX_TEST(!fi.is_ready());
    HPX_TEST(!fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::deferred);
}

///////////////////////////////////////////////////////////////////////////////
void test_invoking_a_packaged_task_populates_future()
{
    hpx::lcos::local::packaged_task<int()> pt(make_int);
    hpx::lcos::future<int> fi = pt.get_future();

    pt();

    HPX_TEST(fi.is_ready());
    HPX_TEST(fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::ready);

    int i = fi.get();
    HPX_TEST_EQ(i, 42);
}

///////////////////////////////////////////////////////////////////////////////
void test_invoking_a_packaged_task_twice_throws()
{
    hpx::lcos::local::packaged_task<int()> pt(make_int);

    pt();
    try {
        pt();
        HPX_TEST(false);
    }
    catch (hpx::exception const& e) {
        HPX_TEST(e.get_error() == hpx::task_already_started);
    }
    catch (...) {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_cannot_get_future_twice_from_task()
{
    hpx::lcos::local::packaged_task<int()> pt(make_int);
    pt.get_future();
    try {
        pt.get_future();
        HPX_TEST(false);
    }
    catch (hpx::exception const& e) {
        HPX_TEST(e.get_error() == hpx::future_already_retrieved);
    }
    catch (...) {
        HPX_TEST(false);
    }
}

void test_task_stores_exception_if_function_throws()
{
    hpx::lcos::local::packaged_task<int()> pt(throw_runtime_error);
    hpx::lcos::future<int> fi = pt.get_future();

    pt();

    HPX_TEST(fi.is_ready());
    HPX_TEST(!fi.has_value());
    HPX_TEST(fi.has_exception());
    HPX_TEST(fi.get_state()==hpx::lcos::future_status::ready);
    try {
        fi.get();
        HPX_TEST(false);
    }
    catch (std::exception&) {
        HPX_TEST(true);
    }
    catch (...) {
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
    HPX_TEST(f.get_state() == hpx::lcos::future_status::ready);
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
//     HPX_TEST(f.get_state() == hpx::lcos::future_status::ready);
//     HPX_TEST_EQ(&f.get(), &i);
// }

void do_nothing()
{
}

void test_task_returning_void()
{
    hpx::lcos::local::packaged_task<void()> pt(do_nothing);
    hpx::lcos::future<void> fi = pt.get_future();

    pt();

    HPX_TEST(fi.is_ready());
    HPX_TEST(fi.has_value());
    HPX_TEST(!fi.has_exception());
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::ready);
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
//     HPX_TEST(fi.get_state() == hpx::lcos::future_status::ready);
//     int& i = fi.get();
//     HPX_TEST_EQ(&i, &global_ref_target);
// }

void test_shared_future()
{
    hpx::lcos::local::packaged_task<int()> pt(make_int);
    hpx::lcos::future<int> fi = pt.get_future();

    hpx::lcos::future<int> sf(boost::move(fi));
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::uninitialized);

    pt();

    HPX_TEST(sf.is_ready());
    HPX_TEST(sf.has_value());
    HPX_TEST(!sf.has_exception());
    HPX_TEST(sf.get_state() == hpx::lcos::future_status::ready);

    int i = sf.get();
    HPX_TEST_EQ(i, 42);
}

void test_copies_of_shared_future_become_ready_together()
{
    hpx::lcos::local::packaged_task<int()> pt(make_int);
    hpx::lcos::future<int> fi=pt.get_future();

    hpx::lcos::future<int> sf1(boost::move(fi));
    hpx::lcos::future<int> sf2(sf1);
    hpx::lcos::future<int> sf3;

    sf3 = sf1;
    HPX_TEST(sf1.get_state() == hpx::lcos::future_status::deferred);
    HPX_TEST(sf2.get_state() == hpx::lcos::future_status::deferred);
    HPX_TEST(sf3.get_state() == hpx::lcos::future_status::deferred);

    pt();

    HPX_TEST(sf1.is_ready());
    HPX_TEST(sf1.has_value());
    HPX_TEST(!sf1.has_exception());
    HPX_TEST(sf1.get_state() == hpx::lcos::future_status::ready);
    int i = sf1.get();
    HPX_TEST_EQ(i, 42);

    i = 0;
    HPX_TEST(sf2.is_ready());
    HPX_TEST(sf2.has_value());
    HPX_TEST(!sf2.has_exception());
    HPX_TEST(sf2.get_state() == hpx::lcos::future_status::ready);
    i = sf2.get();
    HPX_TEST_EQ(i, 42);

    i = 0;
    HPX_TEST(sf3.is_ready());
    HPX_TEST(sf3.has_value());
    HPX_TEST(!sf3.has_exception());
    HPX_TEST(sf3.get_state() == hpx::lcos::future_status::ready);
    i = sf3.get();
    HPX_TEST_EQ(i, 42);
}

void test_shared_future_can_be_move_assigned_from_unique_future()
{
    hpx::lcos::local::packaged_task<int()> pt(make_int);
    hpx::lcos::future<int> fi=pt.get_future();

    hpx::lcos::future<int> sf;
    sf = boost::move(fi);
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::uninitialized);

    HPX_TEST(!sf.is_ready());
    HPX_TEST(!sf.has_value());
    HPX_TEST(!sf.has_exception());
    HPX_TEST(sf.get_state() == hpx::lcos::future_status::deferred);
}

void test_shared_future_void()
{
    hpx::lcos::local::packaged_task<void()> pt(do_nothing);
    hpx::lcos::future<void> fi = pt.get_future();

    hpx::lcos::future<void> sf(boost::move(fi));
    HPX_TEST(fi.get_state() == hpx::lcos::future_status::uninitialized);

    pt();

    HPX_TEST(sf.is_ready());
    HPX_TEST(sf.has_value());
    HPX_TEST(!sf.has_exception());
    HPX_TEST(sf.get_state() == hpx::lcos::future_status::ready);
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
//     HPX_TEST(f.get_state() == hpx::lcos::future_status::ready);
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

hpx::lcos::local::mutex callback_mutex;
unsigned callback_called = 0;

void wait_callback(hpx::lcos::future<int>)
{
    boost::lock_guard<hpx::lcos::local::mutex> lk(callback_mutex);
    ++callback_called;
}

void promise_set_value(hpx::lcos::local::promise<int>& pi)
{
    try {
        pi.set_value(42);
    }
    catch (...) {
    }
}

void test_wait_callback()
{
    callback_called = 0;
    hpx::lcos::local::promise<int> pi;
    hpx::lcos::future<int> fi = pi.get_future();

    fi.then(&wait_callback);
    hpx::thread t(hpx::util::bind(&promise_set_value, boost::ref(pi)));

    fi.wait();

    t.join();

    HPX_TEST_EQ(callback_called, 1U);
    HPX_TEST_EQ(fi.get(), 42);
    fi.wait();
    fi.wait();
    HPX_TEST_EQ(callback_called, 1U);
}

// void do_nothing_callback(hpx::lcos::local::promise<int>& /*pi*/)
// {
//     boost::lock_guard<hpx::lcos::local::mutex> lk(callback_mutex);
//     ++callback_called;
// }
//
// void test_wait_callback_with_timed_wait()
// {
//     callback_called = 0;
//     hpx::lcos::local::promise<int> pi;
//     hpx::lcos::future<int> fi = pi.get_future();
//
//     fi.then(hpx::util::bind(&do_nothing_callback, boost::ref(pi)));
//
//     BOOST_SCOPED_ENUM(hpx::lcos::future_status) state =
//         fi.wait_for(boost::posix_time::milliseconds(10));
//     HPX_TEST_EQ(state, hpx::lcos::future_status::timeout);
//     HPX_TEST_EQ(callback_called, 0U);
//
//     state = fi.wait_for(boost::posix_time::milliseconds(10));
//     HPX_TEST_EQ(state, hpx::lcos::future_status::timeout);
//     state = fi.wait_for(boost::posix_time::milliseconds(10));
//     HPX_TEST_EQ(state, hpx::lcos::future_status::timeout);
//     HPX_TEST_EQ(callback_called, 0U);
//
//     pi.set_value(42);
//
//     state = fi.wait_for(boost::posix_time::milliseconds(10));
//     HPX_TEST_EQ(state, hpx::lcos::future_status::ready);
//
//     HPX_TEST_EQ(callback_called, 1U);
//     HPX_TEST_EQ(fi.get(), 42);
//     HPX_TEST_EQ(callback_called, 1U);
// }

// void wait_callback_for_task(hpx::lcos::local::packaged_task<int()>& pt)
// {
//     boost::lock_guard<hpx::lcos::local::mutex> lk(callback_mutex);
//     ++callback_called;
//     try {
//         pt();
//     }
//     catch (...) {
//     }
// }

// void test_wait_callback_for_packaged_task()
// {
//     callback_called=0;
//     hpx::lcos::local::packaged_task<int()> pt(make_int);
//     hpx::lcos::future<int> fi = pt.get_future();
//     pt.set_wait_callback(wait_callback_for_task);
//     fi.wait();
//     HPX_TEST(callback_called);
//     HPX_TEST_EQ(fi.get(), 42);
//     fi.wait();
//     fi.wait();
//     HPX_TEST_EQ(callback_called, 1U);
// }

void test_packaged_task_can_be_moved()
{
    hpx::lcos::local::packaged_task<int()> pt(make_int);
    hpx::lcos::future<int> fi = pt.get_future();
    HPX_TEST(!fi.is_ready());

    hpx::lcos::local::packaged_task<int()> pt2(boost::move(pt));
    HPX_TEST(!fi.is_ready());

    try {
        pt();
        HPX_TEST(!"Can invoke moved task!");
    }
    catch (hpx::exception const& e) {
      HPX_TEST(e.get_error() == hpx::task_moved);
    }
    catch (...) {
        HPX_TEST(false);
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
        HPX_TEST(false);    // shouldn't get here
    }
    catch (hpx::exception const& e) {
        HPX_TEST(e.get_error() == hpx::broken_promise);
    }
    catch (...) {
        HPX_TEST(false);
    }
}

void test_destroying_a_packaged_task_stores_broken_task()
{
    hpx::lcos::future<int> f;

    {
        hpx::lcos::local::packaged_task<int()> p(make_int);
        f = p.get_future();
    }

    HPX_TEST(f.is_ready());
    HPX_TEST(f.has_exception());
    try {
        f.get();
        HPX_TEST(false);    // shouldn't get here
    }
    catch (hpx::exception const& e) {
      HPX_TEST(e.get_error() == hpx::broken_task);
    }
    catch (...) {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
int make_int_slowly()
{
    hpx::this_thread::sleep_for(boost::posix_time::milliseconds(100));
    return 42;
}

void test_wait_for_either_of_two_futures_1()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());

    pt1();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 0);
    HPX_TEST(f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST_EQ(f1.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_two_futures_2()
{
    hpx::lcos::local::packaged_task<int()> pt(make_int_slowly);
    hpx::lcos::future<int> f1(pt.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());

    pt2();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 1);
    HPX_TEST(!f1.is_ready());
    HPX_TEST(f2.is_ready());
    HPX_TEST_EQ(f2.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_two_futures_list_1()
{
    std::vector<hpx::lcos::future<int> > futures;
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    futures.push_back(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    futures.push_back(pt2.get_future());

    pt1();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(futures);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 0);
    HPX_TEST(futures[0].is_ready());
    HPX_TEST(!futures[1].is_ready());
    HPX_TEST_EQ(futures[0].get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_two_futures_list_2()
{
    std::vector<hpx::lcos::future<int> > futures;
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    futures.push_back(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    futures.push_back(pt2.get_future());

    pt2();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(futures);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 1);
    HPX_TEST(!futures[0].is_ready());
    HPX_TEST(futures[1].is_ready());
    HPX_TEST_EQ(futures[1].get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_three_futures_1()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());

    pt1();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 0);
    HPX_TEST(f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST(!f3.is_ready());
    HPX_TEST_EQ(f1.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_three_futures_2()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());

    pt2();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 1);
    HPX_TEST(!f1.is_ready());
    HPX_TEST(f2.is_ready());
    HPX_TEST(!f3.is_ready());
    HPX_TEST_EQ(f2.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_three_futures_3()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());

    pt3();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 2);
    HPX_TEST(!f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST(f3.is_ready());
    HPX_TEST_EQ(f3.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_four_futures_1()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4(pt4.get_future());

    pt1();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3, f4);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 0);
    HPX_TEST(f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST(!f3.is_ready());
    HPX_TEST(!f4.is_ready());
    HPX_TEST_EQ(f1.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_four_futures_2()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4(pt4.get_future());

    pt2();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3, f4);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 1);
    HPX_TEST(!f1.is_ready());
    HPX_TEST(f2.is_ready());
    HPX_TEST(!f3.is_ready());
    HPX_TEST(!f4.is_ready());
    HPX_TEST_EQ(f2.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_four_futures_3()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4(pt4.get_future());

    pt3();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3, f4);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 2);
    HPX_TEST(!f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST(f3.is_ready());
    HPX_TEST(!f4.is_ready());
    HPX_TEST_EQ(f3.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_four_futures_4()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4(pt4.get_future());

    pt4();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3, f4);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 3);
    HPX_TEST(!f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST(!f3.is_ready());
    HPX_TEST(f4.is_ready());
    HPX_TEST_EQ(f4.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_five_futures_1()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4(pt4.get_future());
    hpx::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5(pt5.get_future());

    pt1();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3, f4, f5);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 0);
    HPX_TEST(f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST(!f3.is_ready());
    HPX_TEST(!f4.is_ready());
    HPX_TEST(!f5.is_ready());
    HPX_TEST_EQ(f1.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_five_futures_2()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4(pt4.get_future());
    hpx::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5(pt5.get_future());

    pt2();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3, f4, f5);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 1);
    HPX_TEST(!f1.is_ready());
    HPX_TEST(f2.is_ready());
    HPX_TEST(!f3.is_ready());
    HPX_TEST(!f4.is_ready());
    HPX_TEST(!f5.is_ready());
    HPX_TEST_EQ(f2.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_five_futures_3()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4(pt4.get_future());
    hpx::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5(pt5.get_future());

    pt3();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3, f4, f5);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 2);
    HPX_TEST(!f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST(f3.is_ready());
    HPX_TEST(!f4.is_ready());
    HPX_TEST(!f5.is_ready());
    HPX_TEST_EQ(f3.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_five_futures_4()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4(pt4.get_future());
    hpx::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5(pt5.get_future());

    pt4();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3, f4, f5);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 3);
    HPX_TEST(!f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST(!f3.is_ready());
    HPX_TEST(f4.is_ready());
    HPX_TEST(!f5.is_ready());
    HPX_TEST_EQ(f4.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

void test_wait_for_either_of_five_futures_5()
{
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1(pt1.get_future());
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2(pt2.get_future());
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3(pt3.get_future());
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4(pt4.get_future());
    hpx::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5(pt5.get_future());

    pt5();

    hpx::lcos::future<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > r =
        hpx::when_any(f1, f2, f3, f4, f5);
    HPX_STD_TUPLE<int, hpx::lcos::future<int> > t= r.get();

    HPX_TEST_EQ(HPX_STD_GET(0, t), 4);
    HPX_TEST(!f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST(!f3.is_ready());
    HPX_TEST(!f4.is_ready());
    HPX_TEST(f5.is_ready());
    HPX_TEST_EQ(f5.get(), 42);

    HPX_TEST(HPX_STD_GET(1, t).is_ready());
    HPX_TEST_EQ(HPX_STD_GET(1, t).get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
// void test_wait_for_either_invokes_callbacks()
// {
//     callback_called = 0;
//     hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
//     hpx::lcos::future<int> fi = pt1.get_future();
//     hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
//     hpx::lcos::future<int> fi2 = pt2.get_future();
//     pt1.set_wait_callback(wait_callback_for_task);
//
//     hpx::thread t(boost::move(pt));
//
//     boost::wait_for_any(fi, fi2);
//     HPX_TEST_EQ(callback_called, 1U);
//     HPX_TEST_EQ(fi.get(), 42);
// }

// void test_wait_for_any_from_range()
// {
//     unsigned const count = 10;
//     for(unsigned i = 0; i < count; ++i)
//     {
//         hpx::lcos::local::packaged_task<int()> tasks[count];
//         hpx::lcos::future<int> futures[count];
//         for(unsigned j = 0; j < count; ++j)
//         {
//             tasks[j] = boost::move(hpx::lcos::local::packaged_task<int()>(make_int_slowly));
//             futures[j] = tasks[j].get_future();
//         }
//         hpx::thread t(boost::move(tasks[i]));
//
//         hpx::lcos::wait_any(futures, futures);
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

void test_wait_for_all_from_list()
{
    unsigned const count = 10;
    std::vector<hpx::lcos::future<int> > futures;
    for (unsigned j = 0; j < count; ++j)
    {
        hpx::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures.push_back(task.get_future());
        task.apply();
    }

    hpx::lcos::future<std::vector<hpx::lcos::future<int> > > r =
        hpx::when_all(futures);

    std::vector<hpx::lcos::future<int> > result = r.get();

    HPX_TEST_EQ(futures.size(), result.size());
    for (unsigned j = 0; j < count; ++j)
    {
        HPX_TEST(futures[j].is_ready());
        HPX_TEST(result[j].is_ready());
    }
}

void test_wait_for_all_two_futures()
{
    unsigned const count = 2;

    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    pt2.apply();

    hpx::lcos::future<std::vector<hpx::lcos::future<int> > > r =
        hpx::when_all(f1, f2);

    std::vector<hpx::lcos::future<int> > result = r.get();

    HPX_TEST_EQ(result.size(), count);
    for (unsigned j = 0; j < count; ++j)
    {
        HPX_TEST(result[j].is_ready());
    }
    HPX_TEST(f1.is_ready());
    HPX_TEST(f2.is_ready());
}

void test_wait_for_all_three_futures()
{
    unsigned const count = 3;

    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    pt2.apply();
    hpx::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    pt3.apply();

    hpx::lcos::future<std::vector<hpx::lcos::future<int> > > r =
        hpx::when_all(f1, f2, f3);

    std::vector<hpx::lcos::future<int> > result = r.get();

    HPX_TEST_EQ(result.size(), count);
    for (unsigned j = 0; j < count; ++j)
    {
        HPX_TEST(result[j].is_ready());
    }
    HPX_TEST(f1.is_ready());
    HPX_TEST(f2.is_ready());
    HPX_TEST(f3.is_ready());
}

void test_wait_for_all_four_futures()
{
    unsigned const count = 4;

    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    pt2.apply();
    hpx::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    pt3.apply();
    hpx::lcos::local::futures_factory<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4 = pt4.get_future();
    pt4.apply();

    hpx::lcos::future<std::vector<hpx::lcos::future<int> > > r =
        hpx::when_all(f1, f2, f3, f4);

    std::vector<hpx::lcos::future<int> > result = r.get();

    HPX_TEST_EQ(result.size(), count);
    for (unsigned j = 0; j < count; ++j)
    {
        HPX_TEST(result[j].is_ready());
    }
    HPX_TEST(f1.is_ready());
    HPX_TEST(f2.is_ready());
    HPX_TEST(f3.is_ready());
    HPX_TEST(f4.is_ready());
}

void test_wait_for_all_five_futures()
{
    unsigned const count = 5;

    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    pt2.apply();
    hpx::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    pt3.apply();
    hpx::lcos::local::futures_factory<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4 = pt4.get_future();
    pt4.apply();
    hpx::lcos::local::futures_factory<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5 = pt5.get_future();
    pt5.apply();

    hpx::lcos::future<std::vector<hpx::lcos::future<int> > > r =
        hpx::when_all(f1, f2, f3, f4, f5);

    std::vector<hpx::lcos::future<int> > result = r.get();

    HPX_TEST_EQ(result.size(), count);
    for (unsigned j = 0; j < count; ++j)
    {
        HPX_TEST(result[j].is_ready());
    }
    HPX_TEST(f1.is_ready());
    HPX_TEST(f2.is_ready());
    HPX_TEST(f3.is_ready());
    HPX_TEST(f4.is_ready());
    HPX_TEST(f5.is_ready());
}

void test_wait_for_two_out_of_five_futures()
{
    unsigned const count = 2;

    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    pt2();
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4 = pt4.get_future();
    pt4();
    hpx::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5 = pt5.get_future();

    typedef std::vector<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > result_type;
    hpx::lcos::future<result_type> r = hpx::when_n(count, f1, f2, f3, f4, f5);

    result_type result = r.get();

    HPX_TEST_EQ(result.size(), count);
    for (unsigned j = 0; j < count; ++j)
    {
        HPX_TEST(HPX_STD_GET(1, result[j]).is_ready());
    }
    HPX_TEST(!f1.is_ready());
    HPX_TEST(f2.is_ready());
    HPX_TEST(!f3.is_ready());
    HPX_TEST(f4.is_ready());
    HPX_TEST(!f5.is_ready());

    HPX_TEST_EQ(HPX_STD_GET(0, result[0]), 1);
    HPX_TEST_EQ(HPX_STD_GET(0, result[1]), 3);
}

void test_wait_for_three_out_of_five_futures()
{
    unsigned const count = 3;

    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1();
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    hpx::lcos::local::packaged_task<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    pt3();
    hpx::lcos::local::packaged_task<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4 = pt4.get_future();
    hpx::lcos::local::packaged_task<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5 = pt5.get_future();
    pt5();

    typedef std::vector<HPX_STD_TUPLE<int, hpx::lcos::future<int> > > result_type;
    hpx::lcos::future<result_type> r = hpx::when_n(count, f1, f2, f3, f4, f5);

    result_type result = r.get();

    HPX_TEST_EQ(result.size(), count);
    for (unsigned j = 0; j < count; ++j)
    {
        HPX_TEST(HPX_STD_GET(1, result[j]).is_ready());
    }
    HPX_TEST(f1.is_ready());
    HPX_TEST(!f2.is_ready());
    HPX_TEST(f3.is_ready());
    HPX_TEST(!f4.is_ready());
    HPX_TEST(f5.is_ready());

    HPX_TEST_EQ(HPX_STD_GET(0, result[0]), 0);
    HPX_TEST_EQ(HPX_STD_GET(0, result[1]), 2);
    HPX_TEST_EQ(HPX_STD_GET(0, result[2]), 4);
}

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
        test_set_value_updates_future_status();
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
        test_wait_callback();
//         test_wait_callback_with_timed_wait();
//         test_wait_callback_for_packaged_task();
        test_packaged_task_can_be_moved();
        test_destroying_a_promise_stores_broken_promise();
        test_destroying_a_packaged_task_stores_broken_task();
        test_wait_for_either_of_two_futures_1();
        test_wait_for_either_of_two_futures_2();
        test_wait_for_either_of_two_futures_list_1();
        test_wait_for_either_of_two_futures_list_2();
        test_wait_for_either_of_three_futures_1();
        test_wait_for_either_of_three_futures_2();
        test_wait_for_either_of_three_futures_3();
        test_wait_for_either_of_four_futures_1();
        test_wait_for_either_of_four_futures_2();
        test_wait_for_either_of_four_futures_3();
        test_wait_for_either_of_four_futures_4();
        test_wait_for_either_of_five_futures_1();
        test_wait_for_either_of_five_futures_2();
        test_wait_for_either_of_five_futures_3();
        test_wait_for_either_of_five_futures_4();
        test_wait_for_either_of_five_futures_5();
//         test_wait_for_either_invokes_callbacks();
//         test_wait_for_any_from_range();
        test_wait_for_all_from_list();
        test_wait_for_all_two_futures();
        test_wait_for_all_three_futures();
        test_wait_for_all_four_futures();
        test_wait_for_all_five_futures();
        test_wait_for_two_out_of_five_futures();
        test_wait_for_three_out_of_five_futures();
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

