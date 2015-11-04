// Copyright (C) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/util/lightweight_test.hpp>

int global;

int& foo(){ return global; }

void test_make_ready_future()
{
    hpx::future<int&> f =
        hpx::make_ready_future(boost::ref(global));
    HPX_TEST(&f.get() == &global);

    hpx::future<int&> f_at =
        hpx::make_ready_future_at(
            boost::chrono::system_clock::now() + boost::chrono::seconds(1)
          , boost::ref(global));
    HPX_TEST(&f_at.get() == &global);

    hpx::future<int&> f_after =
        hpx::make_ready_future_after(
            boost::chrono::seconds(1)
          , boost::ref(global));
    HPX_TEST(&f_after.get() == &global);
}

void test_async()
{
    hpx::future<int&> f = hpx::async(&foo);
    HPX_TEST(&f.get() == &global);

    hpx::future<int&> f_sync = hpx::async(hpx::launch::sync, &foo);
    HPX_TEST(&f_sync.get() == &global);
}

int main()
{
    test_make_ready_future();
    test_async();

    return hpx::util::report_errors();
}
