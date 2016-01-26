//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/include/iostreams.hpp>

#include <hpx/util/lightweight_test.hpp>

#include "server_1950.hpp"

int hpx_main(boost::program_options::variables_map &vm) {
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
    boost::program_options::options_description
        desc_commandline("USAGE: " HPX_APPLICATION_STRING " [options]");

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv), 0,
        "HPX exited with exit status != 0");
}
