//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/parallel/spmd_block.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>

std::size_t num_images = 1000;

int main()
{
    using hpx::parallel::execution::par;

    auto bulk_test = [](hpx::parallel::v2::spmd_block block)
                     {
                        HPX_TEST_EQ( block.get_num_images(), num_images );
                        HPX_TEST_EQ( block.this_image() < num_images, true );
                     };

    hpx::parallel::v2::define_spmd_block(par, "block1", num_images, bulk_test);

    return 0;
}
