//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/lcos/broadcast.hpp>
#include <hpx/parallel/execution.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <string>
#include <vector>

#include <boost/atomic.hpp>
#include <boost/range/irange.hpp>

boost::atomic<std::size_t> local_count;

struct spmd_block_helper
{
    std::string name_;
    std::size_t num_images_;

    template <typename ... Ts>
    void operator()(std::size_t image_id, Ts && ... ts) const
    {
        // Ensure that other images reaches that point
        hpx::lcos::barrier barrier(name_ + "_barrier" , num_images_, image_id);
        barrier.wait();

        ++local_count;
    }
};

std::size_t bulk_test(
    std::string name,
    std::size_t images_per_locality,
    std::size_t num_images)
{
    std::size_t offset = hpx::get_locality_id();
    offset *= images_per_locality;

    local_count.store(0);

    hpx::parallel::execution::sync_bulk_execute(
        hpx::parallel::execution::par.executor(),
        spmd_block_helper{name, num_images},
        boost::irange(offset, offset + images_per_locality));

    return local_count.load();
}
HPX_PLAIN_ACTION(bulk_test, bulk_test_action);

int main()
{
    std::string name = "test";
    std::size_t images_per_locality = 4;
    std::size_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::size_t num_images = num_localities * images_per_locality;

    bulk_test_action act;

    std::vector<std::size_t> result = hpx::lcos::broadcast(
            act, hpx::find_all_localities(), name,
            images_per_locality, num_images
        ).get();

    for (std::size_t s : result)
    {
        HPX_TEST_EQ(s, images_per_locality);
    }
    HPX_TEST_EQ(num_localities, result.size());

    return hpx::util::report_errors();
}
