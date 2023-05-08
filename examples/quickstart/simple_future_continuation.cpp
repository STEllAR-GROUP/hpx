//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/runtime.hpp>
#include <hpx/thread.hpp>

#include <iostream>

int get_id(int i)
{
    return i;
}

int func1()
{
    std::cout << "func1 thread id: " << hpx::this_thread::get_id() << std::endl;
    return get_id(1) ? 123 : 0;
}

// this continuation function will be executed by an HPX thread
int cont1(hpx::future<int> f)
{
    std::cout << "cont1 thread id: " << hpx::this_thread::get_id() << std::endl;
    std::cout << "Status code (HPX thread): " << f.get() << std::endl;
    std::cout << std::flush;
    return 1;
}

// this continuation function will be executed by the UI (main) thread, which is
// not an HPX thread
int cont2(hpx::future<int> f)
{
    std::cout << "Status code (main thread): " << f.get() << std::endl;
    return 1;
}

int hpx_main()
{
    // executing continuation cont1 on same thread as func1
    {
        hpx::future<int> t = hpx::async(&func1);
        hpx::future<int> t2 = t.then(hpx::launch::sync, &cont1);
        t2.get();
    }

    // executing continuation cont1 on new HPX thread
    {
        hpx::future<int> t = hpx::async(&func1);
        hpx::future<int> t2 = t.then(hpx::launch::async, &cont1);
        t2.get();
    }

    // executing continuation cont2 on UI (main) thread
    {
        hpx::parallel::execution::main_pool_executor exec;
        hpx::future<int> t = hpx::async(&func1);
        hpx::future<int> t2 = t.then(exec, &cont2);
        t2.get();
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);
}
