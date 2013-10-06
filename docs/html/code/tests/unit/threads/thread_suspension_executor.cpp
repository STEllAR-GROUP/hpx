// Copyright (C) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/lexical_cast.hpp>

void test_executor_association()
{
    hpx::threads::thread_id_type id = hpx::threads::get_self_id();
    hpx::threads::executor exec_before = hpx::threads::get_executor(id);

    hpx::this_thread::suspend();

    hpx::threads::executor exec_after = hpx::threads::get_executor(id);
    HPX_TEST_EQ(exec_before, exec_after);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        hpx::threads::executors::local_priority_queue_executor exec(1);
        hpx::async(exec, &test_executor_association).get();
    }

    {
        hpx::threads::executors::local_priority_queue_executor exec(1);
        hpx::apply(exec, &test_executor_association);
    }
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(0, hpx::init(argc, argv), "hpx::init returned non-zero value");
    return hpx::util::report_errors();
}

