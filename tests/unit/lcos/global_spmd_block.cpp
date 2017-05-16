//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/spmd_block.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <utility>

std::size_t images_per_locality = 4;
std::size_t iterations = 20;
boost::atomic<std::size_t> c(0);

void bulk_test_function(hpx::lcos::spmd_block block)
{
    std::size_t num_images
        = hpx::get_num_localities(hpx::launch::sync) * images_per_locality;

    HPX_TEST_EQ(block.get_num_images(), num_images);
    HPX_TEST_EQ(block.this_image() < num_images, true);

    for(std::size_t i=0, test_count = images_per_locality;
        i<iterations;
        i++, test_count+=images_per_locality)
    {
        ++c;
        block.sync_all();
        HPX_TEST_EQ(c, test_count);
        block.sync_all();
    }
}
HPX_PLAIN_ACTION(bulk_test_function, bulk_test_action);


int main()
{
    bulk_test_action act;

    hpx::future<void> join =
        hpx::lcos::define_spmd_block(
            "block1", images_per_locality, std::move(act));

    hpx::wait_all(join);

    return 0;
}
