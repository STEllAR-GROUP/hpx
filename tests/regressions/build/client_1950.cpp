//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/iostream.hpp>

#include <hpx/modules/testing.hpp>

#include "server_1950.hpp"

int hpx_main(hpx::program_options::variables_map &vm) {
  {
    hpx::cout << "Hello World!\n" << hpx::flush;

    hpx::id_type id = hpx::new_<test_server>(hpx::find_here()).get();
    hpx::future<void> f = hpx::async(call_action(), id);
    f.get();
    HPX_TEST(test_server::called);
  }
  return hpx::finalize();
}

int main(int argc, char **argv)
{
    hpx::program_options::options_description
        desc_commandline("USAGE: " HPX_APPLICATION_STRING " [options]");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX exited with exit status != 0");
}
#endif
