//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/parallel/spmd_block.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>

std::size_t num_images = 10;

int main()
{
    auto bulk_test = [](hpx::parallel::v2::spmd_block block)
                     {
                        std::cout<< "Welcome in image "
                            << block.this_image() << std::endl;

                        HPX_TEST_EQ( block.get_num_images(), num_images );
                        HPX_TEST_EQ( block.this_image() < num_images, true );
                        block.sync_all();
                     };

    hpx::parallel::v2::define_spmd_block("block1", num_images, bulk_test);

    return 0;
}
