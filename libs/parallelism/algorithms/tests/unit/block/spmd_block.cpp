//  Copyright (c) 2017 Antoine Tran Tan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/executors/execution_policy.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/spmd_block.hpp>
#include <hpx/runtime_local/custom_exception_info.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <set>
#include <utility>
#include <vector>

std::size_t num_images = 10;
std::size_t iterations = 20;

void bulk_test_function(
    hpx::parallel::spmd_block block, std::atomic<std::size_t>* c)
{
    HPX_TEST_EQ(block.get_num_images(), num_images);
    HPX_TEST_EQ(block.this_image() < num_images, true);

    // Test sync_all()
    for (std::size_t i = 0, test_count = num_images; i < iterations;
         i++, test_count += num_images)
    {
        ++c[0];
        block.sync_all();
        HPX_TEST_EQ(c[0], test_count);
        block.sync_all();
    }

    // Test sync_images() with individual values
    std::size_t image_id = block.this_image();

    if ((image_id == 0) || (image_id == 1))
    {
        ++c[1];
    }
    block.sync_images(0, 1);
    if ((image_id == 0) || (image_id == 1))
    {
        HPX_TEST_EQ(c[1], (std::size_t) 2);
    }

    if ((image_id == 2) || (image_id == 3) || (image_id == 4))
    {
        ++c[2];
    }
    block.sync_images(2, 3, 4);
    if ((image_id == 2) || (image_id == 3) || (image_id == 4))
    {
        HPX_TEST_EQ(c[2], (std::size_t) 3);
    }

    // Test sync_images() with vector of values
    std::vector<std::size_t> vec_images = {5, 6, 7, 8};

    if ((image_id == 5) || (image_id == 6) || (image_id == 7) ||
        (image_id == 8))
    {
        ++c[3];
    }
    block.sync_images(vec_images);
    if ((image_id == 5) || (image_id == 6) || (image_id == 7) ||
        (image_id == 8))
    {
        HPX_TEST_EQ(c[3], (std::size_t) 4);
    }
    block.sync_images(vec_images);
    if ((image_id == 5) || (image_id == 6) || (image_id == 7) ||
        (image_id == 8))
    {
        ++c[3];
    }
    block.sync_images(vec_images.begin(), vec_images.end());
    if ((image_id == 5) || (image_id == 6) || (image_id == 7) ||
        (image_id == 8))
    {
        HPX_TEST_EQ(c[3], (std::size_t) 8);
    }
}

int main()
{
    using hpx::execution::par;
    using hpx::execution::task;

    auto bulk_test = [](hpx::parallel::spmd_block block,
                         std::atomic<std::size_t>* c) {
        bulk_test_function(std::move(block), c);
    };

    std::array<std::atomic<std::size_t>, 4> c1, c2, c3;

    for (std::size_t i = 0; i < 4; i++)
    {
        c1[i] = c2[i] = c3[i] = (std::size_t) 0;
    }

    hpx::parallel::define_spmd_block(num_images, bulk_test, c1.data());

    std::vector<hpx::future<void>> join = hpx::parallel::define_spmd_block(
        par(task), num_images, bulk_test, c2.data());

    hpx::wait_all(join);

    hpx::parallel::define_spmd_block(num_images, bulk_test_function, c3.data());

    return 0;
}
