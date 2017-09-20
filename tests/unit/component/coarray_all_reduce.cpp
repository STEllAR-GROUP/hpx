//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/components/containers/coarray/coarray.hpp>
#include <hpx/lcos/spmd_block.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
// coarray<double> is predefined in the partitioned_vector module
HPX_REGISTER_COARRAY_DECLARATION(double);

void bulk_test( hpx::lcos::spmd_block block,
                std::string values)
{
    using hpx::container::placeholders::_;

    hpx::coarray<double,1> a(block, values, {_}, 1);

    double num_images = block.get_num_images();
    double reduced_value = 0.;
    double guessed_value = (0. + num_images - 1.) * num_images / 2.;

    // Each image writes its local value
    a(_)[0] = block.this_image();

    block.sync_all();

    for (std::size_t i = 0; i<num_images; i++)
    {
        // Each image performs the reduction operation
        reduced_value += a(i)[0];
    }

    block.sync_all();

    HPX_TEST_EQ(reduced_value, guessed_value);
}
HPX_PLAIN_ACTION(bulk_test, bulk_test_action);


int main()
{
    std::string values("values");

    hpx::future<void> join =
        hpx::lcos::define_spmd_block("block", 4, bulk_test_action(),
            values);

    hpx::wait_all(join);

    return 0;
}
