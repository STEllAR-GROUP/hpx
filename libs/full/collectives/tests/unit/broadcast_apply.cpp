//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

void f1() {}
HPX_PLAIN_ACTION(f1);

HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION(f1_action)
HPX_REGISTER_BROADCAST_APPLY_ACTION(f1_action)

void f2(std::uint32_t) {}
HPX_PLAIN_ACTION(f2);

HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION(f2_action)
HPX_REGISTER_BROADCAST_APPLY_ACTION(f2_action)

void f1_idx(std::size_t) {}
HPX_PLAIN_ACTION(f1_idx);

HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_DECLARATION(f1_idx_action)
HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION(f1_idx_action)

void f2_idx(std::uint32_t, std::size_t) {}
HPX_PLAIN_ACTION(f2_idx);

HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION_DECLARATION(f2_idx_action)
HPX_REGISTER_BROADCAST_APPLY_WITH_INDEX_ACTION(f2_idx_action)

int hpx_main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    {
        hpx::lcos::broadcast_apply<f1_action>(localities);
        hpx::lcos::broadcast_apply<f2_action>(localities, 0);
    }
    {
        hpx::lcos::broadcast_apply_with_index<f1_idx_action>(localities);
        hpx::lcos::broadcast_apply_with_index<f2_idx_action>(localities, 0);
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
#endif
