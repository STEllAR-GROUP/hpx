// Copyright (c)       2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>

int hpx_main(int argc, char* argv[])
{
    {
        hpx::id_type promise_id;
        {
            hpx::promise<int> p;

            {
                auto local_promise_id = p.get_id();
                hpx::cout << local_promise_id << hpx::endl;
            }

            hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));

            promise_id = p.get_id();
            hpx::cout << promise_id << hpx::endl;
        }

        hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));

        // This segfaults, because the promise is not alive any more.
        // It SHOULD get kept alive by AGAS though.
        hpx::set_lco_value(promise_id, 10, false);

    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // initialize HPX, run hpx_main.
    hpx::start(argc, argv);

    // wait for hpx::finalize being called.
    return hpx::stop();
}
