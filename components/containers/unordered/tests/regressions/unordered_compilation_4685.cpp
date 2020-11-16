//  Copyright (c) 2020 Nick Robison
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/unordered_map.hpp>

#include <string>
#include <vector>

HPX_REGISTER_UNORDERED_MAP(std::string, double);

const char* const map_name = "mobility_visit_map";

int hpx_main()
{
    hpx::unordered_map<std::string, double> m;

    if (0 == hpx::get_locality_id())
    {
        auto localities = hpx::find_all_localities();

        m = hpx::unordered_map<std::string, double>(
            hpx::container_layout(10, localities));
        auto f1 = m.register_as(map_name);
        f1.get();
    }
    else
    {
        auto f1 = m.connect_to(map_name);
        f1.get();
    }

    return hpx::finalize();
}

int main(int argc, char** argv)
{
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
#endif
