//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/parallel/spmd_block.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <functional>
#include <utility>

std::size_t num_images = 10;
std::size_t iterations = 20;

int main()
{
    using hpx::parallel::execution::par;
    using hpx::parallel::execution::task;

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

    auto join =
        hpx::parallel::v2::define_spmd_block(
            par(task),
                num_images, std::move(bulk_test), std::ref(c2));

    hpx::wait_all(join);

    return 0;
}
