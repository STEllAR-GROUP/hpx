//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/lcos/broadcast.hpp>
#include <hpx/parallel/execution_policy.hpp>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>


struct spmd_block_helper
{
    std::string name_;
    std::size_t num_images_;

    template <typename ... Ts>
    void operator()(std::size_t image_id, Ts && ... ts) const
    {
        printf("image_id is equal to %lu and num_images is %lu\n",
            image_id, num_images_);

        std::cout << "The barrier name is "
            << name_ + "_barrier" << std::endl;

        hpx::lcos::barrier
            barrier(name_ + "_barrier" , num_images_, image_id);

        // Ensure that other images reaches that point
        barrier.wait();
    }
};


void bulk_test(
    std::string name,
    std::size_t images_per_locality,
    std::size_t num_images)
{

    using hpx::parallel::execution::par;

    std::size_t offset = hpx::get_locality_id();
    offset *= images_per_locality;

    return
        hpx::parallel::executor_traits<
            decltype(par)::executor_type
            >::bulk_execute(
                par.executor(),
                spmd_block_helper{name,num_images},
                boost::irange(
                    offset, offset + images_per_locality));
}
HPX_PLAIN_ACTION(bulk_test, bulk_test_action);



int main()
{
    std::string name = "test";
    std::size_t images_per_locality = 4;
    std::size_t num_images
        = hpx::get_num_localities(hpx::launch::sync) * images_per_locality;

    bulk_test_action act;

    hpx::util::unwrapped(hpx::lcos::broadcast(
        act, hpx::find_all_localities(), name, images_per_locality, num_images));

    std::cout << "done...\n";

    return 0;
}
