//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/parallel/spmd_block.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

std::size_t num_images = 10;
std::size_t iterations = 20;

int main()
{
    using hpx::parallel::execution::par;
    using hpx::parallel::execution::task;

    // FIXME : the atomic variable is passed by pointer in place of by
    // reference because of some issue related to std::ref when using certain
    // stdlib versions
/*
    auto bulk_test =
        [](hpx::parallel::v2::spmd_block block, boost::atomic<std::size_t> & c)
        {
            HPX_TEST_EQ(block.get_num_images(), num_images);
            HPX_TEST_EQ(block.this_image() < num_images, true);

            for(std::size_t i=0, test_count = num_images;
                i<iterations;
                i++, test_count+=num_images)
            {
                ++c;
                block.sync_all();
                HPX_TEST_EQ(c, test_count);
                block.sync_all();
            }
        };

    boost::atomic<std::size_t> c1(0), c2(0);

    hpx::parallel::v2::define_spmd_block(
        num_images, std::move(bulk_test), std::ref(c1));

    std::vector<hpx::future<void>> join =
        hpx::parallel::v2::define_spmd_block(
            par(task),
                num_images, std::move(bulk_test), std::ref(c2));

    hpx::wait_all(join);
*/

    auto bulk_test =
        [](hpx::parallel::v2::spmd_block block, boost::atomic<std::size_t> * cptr)
        {
            boost::atomic<std::size_t> & c = *cptr;

            HPX_TEST_EQ(block.get_num_images(), num_images);
            HPX_TEST_EQ(block.this_image() < num_images, true);

            for(std::size_t i=0, test_count = num_images;
                i<iterations;
                i++, test_count+=num_images)
            {
                ++c;
                block.sync_all();
                HPX_TEST_EQ(c, test_count);
                block.sync_all();
            }
        };

    boost::atomic<std::size_t> c1(0), c2(0);

    hpx::parallel::v2::define_spmd_block(
        num_images, std::move(bulk_test), &c1);

    std::vector<hpx::future<void>> join =
        hpx::parallel::v2::define_spmd_block(
            par(task),
                num_images, std::move(bulk_test), &c2);

    hpx::wait_all(join);

    return 0;
}
