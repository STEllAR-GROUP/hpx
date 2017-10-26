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
    HPX_TEST_EQ(std::size_t(4), hpx::resource::get_num_threads("default"));

    hpx::threads::detail::thread_pool_base& tp =
        hpx::resource::get_thread_pool("default");

    HPX_TEST_EQ(hpx::threads::count(tp.get_used_processing_units()), std::size_t(4));
//
//     // Check number of used resources
//     tp.remove_processing_unit(0);
//     HPX_TEST_EQ(std::size_t(3), hpx::threads::count(tp.get_used_processing_units()));
//     tp.remove_processing_unit(1);
//     HPX_TEST_EQ(std::size_t(2), hpx::threads::count(tp.get_used_processing_units()));
//     tp.remove_processing_unit(2);
//     HPX_TEST_EQ(std::size_t(1), hpx::threads::count(tp.get_used_processing_units()));
//
//     tp.add_processing_unit(0, 0 + tp.get_thread_offset());
//     HPX_TEST_EQ(std::size_t(2), hpx::threads::count(tp.get_used_processing_units()));
//     tp.add_processing_unit(1, 1 + tp.get_thread_offset());
//     HPX_TEST_EQ(std::size_t(3), hpx::threads::count(tp.get_used_processing_units()));
//     tp.add_processing_unit(2, 2 + tp.get_thread_offset());
//     HPX_TEST_EQ(std::size_t(4), hpx::threads::count(tp.get_used_processing_units()));
//
//     // Check when removing all but one, we end up on the same thread
//     std::size_t num_thread = 0;
//     auto test_function = [&num_thread, &tp]()
//     {
//         HPX_TEST_EQ(num_thread + tp.get_thread_offset(), hpx::get_worker_thread_num());
//     };
//
//     num_thread = 0;
//     tp.remove_processing_unit(1);
//     tp.remove_processing_unit(2);
//     tp.remove_processing_unit(3);
//     hpx::async(test_function).get();
//     tp.add_processing_unit(1, 1 + tp.get_thread_offset());
//     tp.add_processing_unit(2, 2 + tp.get_thread_offset());
//     tp.add_processing_unit(3, 3 + tp.get_thread_offset());
//
//     num_thread = 1;
//     tp.remove_processing_unit(0);
//     tp.remove_processing_unit(2);
//     tp.remove_processing_unit(3);
//     hpx::async(test_function).get();
//     tp.add_processing_unit(0, 0 + tp.get_thread_offset());
//     tp.add_processing_unit(2, 2 + tp.get_thread_offset());
//     tp.add_processing_unit(3, 3 + tp.get_thread_offset());
//
//     num_thread = 2;
//     tp.remove_processing_unit(0);
//     tp.remove_processing_unit(1);
//     tp.remove_processing_unit(3);
//     hpx::async(test_function).get();
//     tp.add_processing_unit(0, 0 + tp.get_thread_offset());
//     tp.add_processing_unit(1, 1 + tp.get_thread_offset());
//     tp.add_processing_unit(3, 3 + tp.get_thread_offset());
//
//     num_thread = 3;
//     tp.remove_processing_unit(0);
//     tp.remove_processing_unit(1);
//     tp.remove_processing_unit(2);
//     hpx::async(test_function).get();
//     tp.add_processing_unit(0, 0 + tp.get_thread_offset());
//     tp.add_processing_unit(1, 1 + tp.get_thread_offset());
//     tp.add_processing_unit(2, 2 + tp.get_thread_offset());
//
//     // Check random scheduling with reducing resources.
//     num_thread = 0;
//     bool up = true;
//     std::vector<hpx::future<void>> fs;
//     hpx::util::high_resolution_timer t;
//     while (t.elapsed() < 2)
//     {
//         for (std::size_t i = 0; i < hpx::resource::get_num_threads("default") * 10;
//              ++i)
//         {
//             fs.push_back(hpx::async([](){}));
//         }
//         if (up)
//         {
//             if (num_thread != hpx::resource::get_num_threads("default") -1)
//             {
//                 tp.remove_processing_unit(num_thread);
//             }
//
//             ++num_thread;
//             if (num_thread == hpx::resource::get_num_threads("default"))
//             {
//                 up = false;
//                 --num_thread;
//             }
//         }
//         else
//         {
//             tp.add_processing_unit(
//                 num_thread - 1, num_thread + tp.get_thread_offset() - 1);
//             --num_thread;
//             if (num_thread == 0)
//             {
//                 up = true;
//             }
//         }
//     }
//     hpx::when_all(std::move(fs)).get();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg = {
        "hpx.os_threads=4"
    };

    // set up the resource partitioner
    hpx::resource::partitioner rp(argc, argv, std::move(cfg));

    // now run the test
    HPX_TEST_EQ(hpx::init(), 0);
    return hpx::util::report_errors();
}
