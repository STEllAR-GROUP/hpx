//  Copyright (c) 2014 Jeremy Kemp
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test illustrates #1127: executor causes segmentation fault

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/thread_executors.hpp>

#include <boost/shared_ptr.hpp>

using hpx::threads::executors::local_priority_queue_executor;

class mock_runtime
{
public:
    boost::shared_ptr<local_priority_queue_executor> exec;
};

boost::shared_ptr<mock_runtime> test_runtime;

int hpx_main()
{
    test_runtime->exec.reset(new local_priority_queue_executor);
    test_runtime->exec.reset();

    return hpx::finalize();
}

int main()
{
    test_runtime.reset(new mock_runtime);
    return hpx::init();
}

