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
    test_environment(char const* desc)
      : desc_(desc), sem_(0), counter1_(0) {}

    ~test_environment()
    { HPX_TEST_EQ(counter1_, 0); }

    std::string desc_;
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
char const* const sem_wait1_desc[] =
{
    "sem_wait1_01", "sem_wait1_02", "sem_wait1_03", "sem_wait1_04",
    "sem_wait1_05", "sem_wait1_06", "sem_wait1_07", "sem_wait1_08",
    "sem_wait1_09", "sem_wait1_10", "sem_wait1_11", "sem_wait1_12",
    "sem_wait1_13", "sem_wait1_14", "sem_wait1_15", "sem_wait1_16",
    "sem_wait1_17", "sem_wait1_18", "sem_wait1_19", "sem_wait1_20",
    "sem_wait1_21", "sem_wait1_22", "sem_wait1_23", "sem_wait1_24",
    "sem_wait1_25", "sem_wait1_26", "sem_wait1_27", "sem_wait1_28",
    "sem_wait1_29", "sem_wait1_30", "sem_wait1_31", "sem_wait1_32",
    "sem_wait1_33", "sem_wait1_34", "sem_wait1_35", "sem_wait1_36",
    "sem_wait1_37", "sem_wait1_38", "sem_wait1_39", "sem_wait1_40",
    "sem_wait1_41", "sem_wait1_42", "sem_wait1_43", "sem_wait1_44",
    "sem_wait1_45", "sem_wait1_46", "sem_wait1_47", "sem_wait1_48",
    "sem_wait1_49", "sem_wait1_50", "sem_wait1_51", "sem_wait1_52",
    "sem_wait1_53", "sem_wait1_54", "sem_wait1_55", "sem_wait1_56",
    "sem_wait1_57", "sem_wait1_58", "sem_wait1_59", "sem_wait1_60",
    "sem_wait1_61", "sem_wait1_62", "sem_wait1_63", "sem_wait1_64",
    "sem_wait1_65", "sem_wait1_66", "sem_wait1_67", "sem_wait1_68",
    "sem_wait1_69", "sem_wait1_70", "sem_wait1_71", "sem_wait1_72",
    "sem_wait1_73", "sem_wait1_74", "sem_wait1_75", "sem_wait1_76",
    "sem_wait1_77", "sem_wait1_78", "sem_wait1_79", "sem_wait1_80"
};

char const* const sem_signal2_desc[] =
{
    "sem_signal2_01", "sem_signal2_02", "sem_signal2_03", "sem_signal2_04",
    "sem_signal2_05", "sem_signal2_06", "sem_signal2_07", "sem_signal2_08",
    "sem_signal2_09", "sem_signal2_10", "sem_signal2_11", "sem_signal2_12",
    "sem_signal2_13", "sem_signal2_14", "sem_signal2_15", "sem_signal2_16",
    "sem_signal2_17", "sem_signal2_18", "sem_signal2_19", "sem_signal2_20",
    "sem_signal2_21", "sem_signal2_22", "sem_signal2_23", "sem_signal2_24",
    "sem_signal2_25", "sem_signal2_26", "sem_signal2_27", "sem_signal2_28",
    "sem_signal2_29", "sem_signal2_30", "sem_signal2_31", "sem_signal2_32",
    "sem_signal2_33", "sem_signal2_34", "sem_signal2_35", "sem_signal2_36",
    "sem_signal2_37", "sem_signal2_38", "sem_signal2_39", "sem_signal2_40",
    "sem_signal2_41", "sem_signal2_42", "sem_signal2_43", "sem_signal2_44",
    "sem_signal2_45", "sem_signal2_46", "sem_signal2_47", "sem_signal2_48",
    "sem_signal2_49", "sem_signal2_50", "sem_signal2_51", "sem_signal2_52",
    "sem_signal2_53", "sem_signal2_54", "sem_signal2_55", "sem_signal2_56",
    "sem_signal2_57", "sem_signal2_58", "sem_signal2_59", "sem_signal2_60",
    "sem_signal2_61", "sem_signal2_62", "sem_signal2_63", "sem_signal2_64",
    "sem_signal2_65", "sem_signal2_66", "sem_signal2_67", "sem_signal2_68",
    "sem_signal2_69", "sem_signal2_70", "sem_signal2_71", "sem_signal2_72",
    "sem_signal2_73", "sem_signal2_74", "sem_signal2_75", "sem_signal2_76",
    "sem_signal2_77", "sem_signal2_78", "sem_signal2_79", "sem_signal2_80"
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map &vm)
{
    std::size_t max_semaphore_value = 0;

    if (vm.count("semaphores"))
        max_semaphore_value = vm["semaphores"].as<std::size_t>();

    if (max_semaphore_value > 80)
        HPX_THROW_EXCEPTION(bad_parameter, 
            "hpx_main", "semaphore count specified is higher than 80");
    
    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
    {
        // create a semaphore, which which we will use to make several threads 
        // waiting for another one
        boost::shared_ptr<test_environment> env1(new test_environment("test1"));
    
        // create the threads which will have to wait on the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind
                (&sem_wait1, env1, max_semaphore_value), sem_wait1_desc[i]);
    
        // now create a thread signaling the semaphore
        register_work(boost::bind
            (&sem_signal1, env1, max_semaphore_value), "sem_signal1");
    
        // create a semaphore, which which we will use to make several threads 
        // waiting for another one, use signal_all
        boost::shared_ptr<test_environment>
            env1_all(new test_environment("test1_all"));
    
        // create the  threads which will have to wait on the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind
                (&sem_wait1, env1_all, max_semaphore_value), sem_wait1_desc[i]);
    
        // now create a thread signaling the semaphore
        register_work(boost::bind
            (&sem_signal1_all, env1_all, max_semaphore_value), "sem_signal1_all");
    
        // create a semaphore, which we will use to make several threads 
        // waiting for another one, but the semaphore is signaled before being 
        // waited on
        boost::shared_ptr<test_environment> env2(new test_environment("test2"));
    
        // create a thread signaling the semaphore
        register_work(boost::bind
            (&sem_signal1, env2, max_semaphore_value), "sem_signal1");
    
        // create the  threads which will have to wait on the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind
                (&sem_wait1, env2, max_semaphore_value), sem_wait1_desc[i]);
    
        // create a semaphore, which we will use to make several threads 
        // waiting for another one, but the semaphore is signaled before being 
        // waited on, use signal_all
        boost::shared_ptr<test_environment>
            env2_all(new test_environment("test2_all"));
    
        // create a thread signaling the semaphore
        register_work(boost::bind
            (&sem_signal1_all, env2_all, max_semaphore_value), "sem_signal1_all");
    
        // create the threads which will have to wait on the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind
                (&sem_wait1, env2_all, max_semaphore_value), sem_wait1_desc[i]);
    
        // the 3rd test does the opposite, it creates a semaphore, which  
        // will be used to make one thread waiting for several other threads
        boost::shared_ptr<test_environment> env3(new test_environment("test3"));
    
        // now create a thread waiting on the semaphore
        register_work(boost::bind
            (&sem_wait2, env3, max_semaphore_value), "sem_wait2");
    
        // create the threads which will have to signal the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind(&sem_signal2, env3), sem_signal2_desc[i]);
    
        // the 3rd test does the opposite, it creates a semaphore, which  
        // will be used to make one thread waiting for several other threads,
        // use wait_all
        boost::shared_ptr<test_environment>
            env3_all(new test_environment("test3_all"));
    
        // now create a thread waiting on the semaphore
        register_work(boost::bind
            (&sem_wait2_all, env3_all, max_semaphore_value), "sem_wait2_all");
    
        // create the threads which will have to signal the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind(&sem_signal2, env3_all), sem_signal2_desc[i]);
    
        // the 4th test does the opposite, it creates a semaphore, which  
        // will be used to make one thread waiting for several other threads, but 
        // the semaphore is signaled before being waited on
        boost::shared_ptr<test_environment> env4(new test_environment("test4"));
    
        // create the threads which will have to signal the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind(&sem_signal2, env4), sem_signal2_desc[i]);
    
        // now create a thread waiting on the semaphore
        register_work(boost::bind
            (&sem_wait2, env4, max_semaphore_value), "sem_wait2");
    
        // the 4th test does the opposite, it creates a semaphore, which  
        // will be used to make one thread waiting for several other threads, but 
        // the semaphore is signaled before being waited on, use wait_all
        boost::shared_ptr<test_environment>
            env4_all(new test_environment("test4_all"));
    
        // create the threads which will have to signal the semaphore
        for (std::size_t i = 0; i < max_semaphore_value; ++i) 
            register_work(boost::bind(&sem_signal2, env4_all), sem_signal2_desc[i]);
    
        // now create a thread waiting on the semaphore
        register_work(boost::bind
            (&sem_wait2_all, env4_all, max_semaphore_value), "sem_wait2_all");
    }

    // initiate shutdown of the runtime system
    finalize();
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
            "the number of semaphores (max 80)")
        ("iterations", value<std::size_t>()->default_value(1 << 6), 
            "the number of times to repeat the test") 
        ;

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

