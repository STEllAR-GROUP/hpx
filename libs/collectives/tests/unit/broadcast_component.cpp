//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

struct test_component1 : hpx::components::component_base<test_component1>
{
    std::uint32_t f1()
    {
        return hpx::get_locality_id();
    }
    HPX_DEFINE_COMPONENT_ACTION(test_component1, f1);
};

using test_component1_type = hpx::components::component<test_component1>;
HPX_REGISTER_COMPONENT(test_component1_type, test_component1_type);

using f1_action = test_component1_type::f1_action;

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(f1_action)
HPX_REGISTER_BROADCAST_ACTION(f1_action)

////////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::id_type here = hpx::find_here();
    std::vector<hpx::id_type> components =
        hpx::new_<test_component1[]>(here, 10).get();

    std::vector<std::uint32_t> f1_res;
    f1_res = hpx::lcos::broadcast<f1_action>(components).get();

    HPX_TEST_EQ(f1_res.size(), components.size());
    for (std::size_t i = 0; i != f1_res.size(); ++i)
    {
        HPX_TEST_EQ(
            f1_res[i], hpx::naming::get_locality_id_from_id(components[i]));
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
