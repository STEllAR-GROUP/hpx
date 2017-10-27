//  Copyright (c) 2017 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying basic resource_partitioner functionality.

#include <hpx/hpx_init.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    std::size_t const num_threads = hpx::resource::get_num_threads("default");

    HPX_ASSERT(num_threads >= 2);

    // TODO: Leave in or remove? Require at least 2 threads?
    //HPX_TEST_EQ(std::size_t(4), hpx::resource::get_num_threads("default"));

    hpx::threads::detail::thread_pool_base& tp =
        hpx::resource::get_thread_pool("default");

    //HPX_TEST_EQ(hpx::threads::count(tp.get_used_processing_units()), std::size_t(4));

    // Check number of used resources
    for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
    {
        HPX_TEST_EQ(std::size_t(num_threads - thread_num),
                    hpx::threads::count(tp.get_used_processing_units()));
        tp.remove_processing_unit(thread_num);
        HPX_TEST_EQ(std::size_t(num_threads - thread_num - 1),
            hpx::threads::count(tp.get_used_processing_units()));
    }

    for (std::size_t thread_num = 0; thread_num < num_threads - 1; ++thread_num)
    {
        HPX_TEST_EQ(std::size_t(thread_num + 1),
                    hpx::threads::count(tp.get_used_processing_units()));
        tp.add_processing_unit(thread_num, thread_num + tp.get_thread_offset());
        HPX_TEST_EQ(std::size_t(thread_num + 2),
            hpx::threads::count(tp.get_used_processing_units()));
    }

    // Check when removing all but one, we end up on the same thread
    std::size_t num_thread = 0;
    auto test_function = [&num_thread, &tp]()
    {
        HPX_TEST_EQ(num_thread + tp.get_thread_offset(),
            hpx::get_worker_thread_num());
    };

    for (num_thread = 0;
        num_thread < num_threads;
        ++num_thread)
    {
        for (std::size_t num_thread_remove = 0;
             num_thread_remove < num_threads;
             ++num_thread_remove)
        {
            if (num_thread != num_thread_remove)
            {
                tp.remove_processing_unit(num_thread_remove);
            }
        }

        hpx::async(test_function).get();

        for (std::size_t num_thread_add = 0;
             num_thread_add < num_threads;
             ++num_thread_add)
        {
            if (num_thread != num_thread_add)
            {
                tp.add_processing_unit(num_thread_add,
                    num_thread_add + tp.get_thread_offset());
            }
        }
    }

    // Check random scheduling with reducing resources.
    std::size_t thread_num = 0;
    bool up = true;
    std::vector<hpx::future<void>> fs;
    hpx::util::high_resolution_timer t;
    while (t.elapsed() < 2)
    {
        for (std::size_t i = 0;
            i < hpx::resource::get_num_threads("default") * 10;
            ++i)
        {
            fs.push_back(hpx::async([](){}));
        }
        if (up)
        {
            if (thread_num != hpx::resource::get_num_threads("default") - 1)
            {
                tp.remove_processing_unit(thread_num);
            }

            ++thread_num;

            if (thread_num == hpx::resource::get_num_threads("default"))
            {
                up = false;
                --thread_num;
            }
        }
        else
        {
            tp.add_processing_unit(
                thread_num - 1, thread_num + tp.get_thread_offset() - 1);

            --thread_num;

            if (thread_num == 0)
            {
                up = true;
            }
        }
    }
    hpx::when_all(std::move(fs)).get();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // TODO: Re-enable fixed number of threads.
    //std::vector<std::string> cfg = {
    //"hpx.os_threads=4"
    //};

    // set up the resource partitioner
    //hpx::resource::partitioner rp(argc, argv, std::move(cfg));

    // now run the test
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
