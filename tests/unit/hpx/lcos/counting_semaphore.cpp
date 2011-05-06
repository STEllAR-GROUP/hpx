//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/atomic.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/lcos/counting_semaphore.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::lcos::counting_semaphore;

using hpx::applier::register_work;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

using hpx::exception;
using hpx::bad_parameter;

///////////////////////////////////////////////////////////////////////////////
struct test_environment
{
    test_environment()
      : sem_(0), counter1_(0) {}

    ~test_environment()
    { HPX_TEST_EQ(counter1_, 0); }

    counting_semaphore sem_;
    boost::atomic<long> counter1_;
};

///////////////////////////////////////////////////////////////////////////////
void sem_wait1(boost::shared_ptr<test_environment> env, int max_semaphore_value)
{
    env->sem_.wait();
    ++env->counter1_;

    // all of the threads need to have incremented the counter
    HPX_TEST(env->counter1_ <= 0);
}

void sem_signal1(boost::shared_ptr<test_environment> env,
                 int max_semaphore_value)
{
    while (--max_semaphore_value >= 0)
    {
        --env->counter1_;
        env->sem_.signal();    // we need to signal all threads
    }
}

void sem_signal1_all(boost::shared_ptr<test_environment> env,
                     int max_semaphore_value)
{
    env->counter1_ -= max_semaphore_value;
    env->sem_.signal(max_semaphore_value); // we need to signal all threads
}

///////////////////////////////////////////////////////////////////////////////
void sem_wait2(boost::shared_ptr<test_environment> env, int max_semaphore_value)
{
    // we wait for the other threads to signal this semaphore
    while (--max_semaphore_value >= 0)
    {
        env->sem_.wait();
        --env->counter1_;
    }

    // all of the threads need to have incremented the counter
    HPX_TEST_EQ(0, env->counter1_);
}

void sem_wait2_all(boost::shared_ptr<test_environment> env,
                   int max_semaphore_value)
{
    // we wait for the other threads to signal this semaphore
    env->sem_.wait(max_semaphore_value);
    env->counter1_ -= max_semaphore_value;

    // all of the threads need to have incremented the counter
    HPX_TEST_EQ(0, env->counter1_);
}

void sem_signal2(boost::shared_ptr<test_environment> env)
{
    ++env->counter1_;
    env->sem_.signal(); // we need to signal the semaphore here
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map &vm)
{
    std::size_t max_semaphore_value = 0;

    if (vm.count("semaphores"))
        max_semaphore_value = vm["semaphores"].as<std::size_t>();
    
    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        // create a semaphore, which which we will use to make several threads 
        // waiting for another one
        boost::shared_ptr<test_environment> env1(new test_environment);
    
        // create the threads which will have to wait on the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind
                (&sem_wait1, env1, max_semaphore_value));
    
        // now create a thread signaling the semaphore
        register_work(boost::bind
            (&sem_signal1, env1, max_semaphore_value));
    
        // create a semaphore, which which we will use to make several threads 
        // waiting for another one, use signal_all
        boost::shared_ptr<test_environment>
            env1_all(new test_environment);
    
        // create the  threads which will have to wait on the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind
                (&sem_wait1, env1_all, max_semaphore_value));
    
        // now create a thread signaling the semaphore
        register_work(boost::bind
            (&sem_signal1_all, env1_all, max_semaphore_value));
    
        // create a semaphore, which we will use to make several threads 
        // waiting for another one, but the semaphore is signaled before being 
        // waited on
        boost::shared_ptr<test_environment> env2(new test_environment);
    
        // create a thread signaling the semaphore
        register_work(boost::bind
            (&sem_signal1, env2, max_semaphore_value));
    
        // create the  threads which will have to wait on the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind
                (&sem_wait1, env2, max_semaphore_value));
    
        // create a semaphore, which we will use to make several threads 
        // waiting for another one, but the semaphore is signaled before being 
        // waited on, use signal_all
        boost::shared_ptr<test_environment>
            env2_all(new test_environment);
    
        // create a thread signaling the semaphore
        register_work(boost::bind
            (&sem_signal1_all, env2_all, max_semaphore_value));
    
        // create the threads which will have to wait on the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind
                (&sem_wait1, env2_all, max_semaphore_value));
    
        // the 3rd test does the opposite, it creates a semaphore, which  
        // will be used to make one thread waiting for several other threads
        boost::shared_ptr<test_environment> env3(new test_environment);
    
        // now create a thread waiting on the semaphore
        register_work(boost::bind
            (&sem_wait2, env3, max_semaphore_value));
    
        // create the threads which will have to signal the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind(&sem_signal2, env3));
    
        // the 3rd test does the opposite, it creates a semaphore, which  
        // will be used to make one thread waiting for several other threads,
        // use wait_all
        boost::shared_ptr<test_environment>
            env3_all(new test_environment);
    
        // now create a thread waiting on the semaphore
        register_work(boost::bind
            (&sem_wait2_all, env3_all, max_semaphore_value));
   
        // create the threads which will have to signal the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind(&sem_signal2, env3_all));
    
        // the 4th test does the opposite, it creates a semaphore, which  
        // will be used to make one thread waiting for several other threads, but 
        // the semaphore is signaled before being waited on
        boost::shared_ptr<test_environment> env4(new test_environment);
    
        // create the threads which will have to signal the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind(&sem_signal2, env4));
    
        // now create a thread waiting on the semaphore
        register_work(boost::bind(&sem_wait2, env4, max_semaphore_value));
    
        // the 4th test does the opposite, it creates a semaphore, which  
        // will be used to make one thread waiting for several other threads, but 
        // the semaphore is signaled before being waited on, use wait_all
        boost::shared_ptr<test_environment>
            env4_all(new test_environment);
    
        // create the threads which will have to signal the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind(&sem_signal2, env4_all));
    
        // now create a thread waiting on the semaphore
        register_work(boost::bind
            (&sem_wait2_all, env4_all, max_semaphore_value));
    }

    // initiate shutdown of the runtime system
    finalize(5.0);
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");
        
    desc_commandline.add_options()
        ("semaphores,s", value<std::size_t>()->default_value(1 << 3), 
            "the number of semaphores")
        ("iterations", value<std::size_t>()->default_value(1), 
            "the number of times to repeat the test") 
        ;

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

