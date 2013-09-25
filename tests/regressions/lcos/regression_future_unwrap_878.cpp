//  Copyright 2013 (c) Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #878: `future::unwrap`
// triggers assertion

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/future.hpp>

int continuation(hpx::future<int>&){ return 0; }

int main()
{
    hpx::promise<hpx::future<int> > promise;
    hpx::future<hpx::future<int> > future = promise.get_future();
    try
    {
        //throw hpx::bad_parameter;
        promise.set_value(42);
    } catch(...) {
        promise.set_exception(boost::current_exception());
    }
    future.unwrap();

    return 0;
}
