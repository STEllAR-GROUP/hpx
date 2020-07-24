//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/modules/testing.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct test_server : hpx::components::component_base<test_server>
{
    test_server(int) {}
};

typedef hpx::components::component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

void test_bulk_new()
{
    auto locs = hpx::find_all_localities();

    std::vector<hpx::id_type> ids =
        hpx::new_<test_server[]>(hpx::default_layout(locs), 10, 42).get();
    (void) ids;

    hpx::future<std::vector<hpx::id_type>> ids_f =
        hpx::new_<test_server[]>(hpx::default_layout(locs), 10, 42);
    (void) ids_f;
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_bulk_new();

    return 0;
}

