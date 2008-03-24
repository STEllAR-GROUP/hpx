/*=============================================================================
    Copyright (c) 2007 Tobias Schwinger
  
    Use modification and distribution are subject to the Boost Software 
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
==============================================================================*/

#include <boost/detail/lightweight_test.hpp>
#include <boost/utility/thread_specific_singleton.hpp>

#include <boost/thread/thread.hpp>

#define LOCKS_PER_THREAD 1000
#define THREADS 10

class X : public boost::thread_specific_singleton<X>
{
    unsigned val_ctor_called;
public:

    explicit X(boost::restricted)
        : val_ctor_called(0xbeef), counter(0) 
    {
        ++n_instances; 
    }

    unsigned counter;

    unsigned f() 
    {
        BOOST_TEST(val_ctor_called == 0xbeef);
        unsigned counter_before_yield = ++counter;
        boost::thread::yield();
        BOOST_TEST(counter == counter_before_yield);
        return 0xbeef;
    }

    static unsigned n_instances; 
};

unsigned X::n_instances = 0;

void test1()
{
    for (int i = 0; i < LOCKS_PER_THREAD; ++i)
        BOOST_TEST(X::instance->f() == 0xbeef);
    BOOST_TEST(X::instance->counter == LOCKS_PER_THREAD);
}

void test2()
{
    X::lease lease;
    for (int i = 0; i < LOCKS_PER_THREAD; ++i)
        BOOST_TEST(lease->f() == 0xbeef);
    BOOST_TEST(lease->counter == LOCKS_PER_THREAD);
}

int main()
{
    BOOST_TEST(! X::n_instances);

    {
        boost::thread_group threads;
        for (int i = 0; i < THREADS; ++i)
            threads.create_thread(&test1);
        threads.join_all();
    }

    BOOST_TEST(X::n_instances == THREADS);

    {
        boost::thread_group threads;
        for (int i = 0; i < THREADS; ++i)
            threads.create_thread(&test2);
        threads.join_all();
    }

    return boost::report_errors();
}

