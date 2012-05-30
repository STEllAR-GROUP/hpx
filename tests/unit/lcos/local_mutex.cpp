//  Copyright (c) 2001-2003 William E. Kempf
//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// TODO: Test timed locks.

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/barrier.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::applier::register_work_nullary;

using hpx::lcos::local::barrier;
using hpx::lcos::local::mutex;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

///////////////////////////////////////////////////////////////////////////////
template <typename M>
struct test_lock
{
    typedef M mutex_type;
    typedef typename M::scoped_lock lock_type;

    void operator()()
    {
        mutex_type mtx;

        // Test the lock's constructors.
        {
            lock_type lock(mtx, boost::defer_lock);
            HPX_TEST(!lock);
        }

        lock_type lock(mtx);
        HPX_TEST(lock ? true : false);

        // Test the lock and unlock methods.
        lock.unlock();
        HPX_TEST(!lock);
        lock.lock();
        HPX_TEST(lock ? true : false);
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename M>
struct test_mutexed_data_lock
{
    typedef M mutex_type;
    typedef typename M::scoped_lock lock_type;

    mutex_type* mtx;
    barrier* barr;
    std::size_t* data;

    test_mutexed_data_lock(mutex_type& m, barrier& b, std::size_t& d)
        : mtx(&m), barr(&b), data(&d) {}

    void operator()() const
    {
        lock_type lock(*mtx);
        HPX_TEST(lock ? true : false);

        ++(*data);
 
        lock.unlock();
        HPX_TEST(!lock);

        barr->wait();
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename M>
struct test_mutexed_data_lock_raii
{
    typedef M mutex_type;
    typedef typename M::scoped_lock lock_type;

    mutex_type* mtx;
    barrier* barr;
    std::size_t* data;

    test_mutexed_data_lock_raii(mutex_type& m, barrier& b, std::size_t& d)
        : mtx(&m), barr(&b), data(&d) {}

    void operator()()
    {
        {
            lock_type lock(*mtx);
            HPX_TEST(lock ? true : false);

            ++(*data);
        }
 
        barr->wait();
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename M>
struct test_try_lock
{
    typedef M mutex_type;
    typedef typename M::scoped_try_lock try_lock_type;

    void operator()() const
    {
        mutex_type mtx;

        // Test the lock's constructors.
        {
            try_lock_type lock(mtx);
            HPX_TEST(lock ? true : false);
        }
        {
            try_lock_type lock(mtx, boost::defer_lock);
            HPX_TEST(!lock);
        }
        try_lock_type lock(mtx);
        HPX_TEST(lock ? true : false);

        // Test the lock, unlock and try_lock methods.
        lock.unlock();
        HPX_TEST(!lock);
        lock.lock();
        HPX_TEST(lock ? true : false);
        lock.unlock();
        HPX_TEST(!lock);
        HPX_TEST(lock.try_lock());
        HPX_TEST(lock ? true : false);
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename M>
struct test_mutexed_data_try_lock
{
    typedef M mutex_type;
    typedef typename M::scoped_try_lock try_lock_type;

    mutex_type* mtx;
    barrier* barr;
    std::size_t* data;

    test_mutexed_data_try_lock(mutex_type& m, barrier& b, std::size_t& d)
        : mtx(&m), barr(&b), data(&d) {}

    void operator()()
    {
        try_lock_type lock(*mtx, boost::defer_lock);
        HPX_TEST(!lock);

        while (!lock.try_lock())
            {}

        HPX_TEST(lock ? true : false);

        ++(*data);

        lock.unlock();
        HPX_TEST(!lock);

        barr->wait();
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename M>
struct test_mutexed_data_try_lock_raii
{
    typedef M mutex_type;
    typedef typename M::scoped_try_lock try_lock_type;

    mutex_type* mtx;
    barrier* barr;
    std::size_t* data;

    test_mutexed_data_try_lock_raii(mutex_type& m, barrier& b, std::size_t& d)
        : mtx(&m), barr(&b), data(&d) {}

    void operator()() const
    {
        {
            try_lock_type lock(*mtx);

            while (!lock)
            {
                lock.try_lock();
            }

            ++(*data);
        }

        barr->wait();
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t threads = 1;

    if (vm.count("threads"))
        threads = vm["threads"].as<std::size_t>();

    std::size_t pxthreads = threads * 8;

    if (vm.count("pxthreads"))
        pxthreads = vm["pxthreads"].as<std::size_t>();
    
    {
        test_lock<mutex> t;

        for (std::size_t i = 0; i < pxthreads; ++i)
            register_work_nullary(HPX_STD_FUNCTION<void()>(t), 
                "test_local_mutex_lock");
    }

    {
        mutex mtx;
        barrier barr(pxthreads + 1);
        std::size_t data = 0;

        test_mutexed_data_lock<mutex> t(mtx, barr, data);
        for (std::size_t i = 0; i < pxthreads; ++i)
            register_work_nullary(HPX_STD_FUNCTION<void()>(t), 
                "test_local_mutex_lock_contention");

        barr.wait();

        mutex::scoped_lock lock(mtx);
        HPX_TEST(lock ? true : false);

        HPX_TEST_EQ(data, pxthreads);
    } 

    {
        mutex mtx;
        barrier barr(pxthreads + 1);
        std::size_t data = 0;

        test_mutexed_data_lock_raii<mutex> t(mtx, barr, data);
        for (std::size_t i = 0; i < pxthreads; ++i)
            register_work_nullary(HPX_STD_FUNCTION<void()>(t), 
                "test_local_mutex_lock_raii_contention");

        barr.wait();

        mutex::scoped_lock lock(mtx);
        HPX_TEST(lock ? true : false);

        HPX_TEST_EQ(data, pxthreads);
    }

    {
        test_try_lock<mutex> t;

        for (std::size_t i = 0; i < pxthreads; ++i)
            register_work_nullary(HPX_STD_FUNCTION<void()>(t), 
                "test_local_mutex_try_lock");
    }

    {
        mutex mtx;
        barrier barr(pxthreads + 1);
        std::size_t data = 0;

        test_mutexed_data_try_lock<mutex> t(mtx, barr, data);
        for (std::size_t i = 0; i < pxthreads; ++i)
            register_work_nullary(HPX_STD_FUNCTION<void()>(t), 
                "test_local_mutex_try_lock_contention");

        barr.wait();

        mutex::scoped_lock lock(mtx);
        HPX_TEST(lock ? true : false);

        HPX_TEST_EQ(data, pxthreads);
    } 

    {
        mutex mtx;
        barrier barr(pxthreads + 1);
        std::size_t data = 0;

        test_mutexed_data_try_lock_raii<mutex> t(mtx, barr, data);
        for (std::size_t i = 0; i < pxthreads; ++i)
            register_work_nullary(HPX_STD_FUNCTION<void()>(t), 
                "test_local_mutex_try_lock_raii_contention");

        barr.wait();

        mutex::scoped_lock lock(mtx);
        HPX_TEST(lock ? true : false);

        HPX_TEST_EQ(data, pxthreads);
    }

    // Initiate shutdown of the runtime system.
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");
    
    desc_commandline.add_options()
        ("pxthreads,T", value<std::size_t>(), 
            "the number of PX threads to invoke for each subtest (default: OS "
            "threads * 8)")
        ;

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::thread::hardware_concurrency());

    // Initialize and run HPX.
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv, cfg), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

