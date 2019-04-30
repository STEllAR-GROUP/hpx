//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/testing.hpp>

int hpx_main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::get_config_entry("hpx.cores", "1"), std::string("3"));
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg =
    {
        "--hpx:cores=3"
    };

    HPX_TEST_EQ(hpx::init(argc, argv, cfg), 0);

    return hpx::util::report_errors();
}
