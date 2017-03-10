//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2014 Martin Stumpf
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>
#include <utility>

namespace mynamespace
{
    void test()
    {
        hpx::cout << "test" << hpx::endl;
    }

    HPX_DEFINE_PLAIN_ACTION(test, test_action);

    static HPX_CONSTEXPR auto t =
        HPX_LAMBDA_ACTION
        []()
        {
            hpx::cout << "test" << hpx::endl;
        };
}

HPX_REGISTER_ACTION(mynamespace::test_action, mynamespace_test_action);

// hpx_main, is the actual main called by hpx
int hpx_main(int argc, char* argv[])
{
    {
        typedef mynamespace::test_action func;
        hpx::async<func>(hpx::find_here());
    }

    // Same test with lambdas
    {
        hpx::async(std::move(mynamespace::t), hpx::find_here());
    }

    // End the program
    return hpx::finalize();
}

// Main, initializes HPX
int main(int argc, char* argv[]){

    // initialize HPX, run hpx_main
    hpx::start(argc, argv);

    // wait for hpx::finalize being called
    return hpx::stop();
}
