//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/spmd_block.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <array>
#include <cstddef>
#include <utility>

std::size_t images_per_locality = 7;
std::size_t iterations = 20;
std::size_t test_value = 4;
std::array<boost::atomic<std::size_t>,4> c;

void bulk_test_function(hpx::lcos::spmd_block block, std::size_t arg)
{
    std::size_t num_images
        = hpx::get_num_localities(hpx::launch::sync) * images_per_locality;

    HPX_TEST_EQ(block.get_images_per_locality(), images_per_locality);
    HPX_TEST_EQ(block.get_num_images(), num_images);
    HPX_TEST_EQ(block.this_image() < num_images, true);
    HPX_TEST_EQ(arg, test_value * 10);

    for(std::size_t i=0, test_count = images_per_locality;
        i<iterations;
        i++, test_count+=images_per_locality)
    {
        ++c[0];
        block.sync_all();
        HPX_TEST_EQ(c[0], test_count);
        block.sync_all();
    }

    // Test sync_images() with individual values
    std::size_t image_id = block.this_image();
    std::size_t o =
        (block.this_image() / images_per_locality) * images_per_locality;

    if((image_id == o+0) || (image_id == o+1))
    {
        ++c[1];
    }
    block.sync_images(o+0,o+1);
    if((image_id == o+0) || (image_id == o+1))
    {
        HPX_TEST_EQ(c[1],(std::size_t)2);
    }

    if((image_id == o+2) || (image_id == o+3))
    {
        ++c[2];
    }
    block.sync_images(o+2,o+3);
    if((image_id == o+2) || (image_id == o+3))
    {
        HPX_TEST_EQ(c[2],(std::size_t)2);
    }

    // Test sync_images() with vector of values
    std::vector<std::size_t> vec_images = {o+4,o+5,o+6};

    if((image_id == o+4) || (image_id == o+5) || (image_id == o+6))
    {
        ++c[3];
    }
    block.sync_images(vec_images);
    if((image_id == o+4) || (image_id == o+5) || (image_id == o+6))
    {
        HPX_TEST_EQ(c[3],(std::size_t)3);
    }
    block.sync_images(vec_images);
    if((image_id == o+4) || (image_id == o+5) || (image_id == o+6))
    {
        ++c[3];
    }
    block.sync_images(vec_images.begin(),vec_images.end());
    if((image_id == o+4) || (image_id == o+5) || (image_id == o+6))
    {
        HPX_TEST_EQ(c[3],(std::size_t)6);
    }
}
HPX_PLAIN_ACTION(bulk_test_function, bulk_test_action);


int main()
{
    std::size_t arg = test_value * 10;
    bulk_test_action act;

    //Initialize our atomics
    for(std::size_t i =0; i<4; i++)
    {
        c[i] = (std::size_t)0;
    }

    hpx::future<void> join =
        hpx::lcos::define_spmd_block(
            "block1", images_per_locality, std::move(act), arg );

    hpx::wait_all(join);

    return 0;
}
